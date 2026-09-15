"""Native Qwen forecasting, optionally conditioned on predicted 4+4 slots.

Current pixels/history remain on Qwen's original multimodal path. Only the
auxiliary state encoder reads frozen, cached JEPA features. Future observations
are accessed exclusively by training losses and the fixed target encoder.
"""
from contextlib import contextmanager
import copy
import math
from pathlib import Path
import sys

import torch
from torch import nn
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
from medworld_common.qwen import load_qwen, masked_bce, chunked_ce

HORIZONS = ['6 to 24 hours', '1 to 3 days', '3 to 7 days', '7 to 30 days']
WIDTH = 1024
LANGUAGE_TARGETS = {'q_proj', 'k_proj', 'v_proj', 'o_proj', 'in_proj_qkv',
                    'in_proj_z', 'in_proj_b', 'in_proj_a', 'out_proj'}


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base.requires_grad_(False)
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.scale = alpha / rank

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x.to(self.lora_a.dtype), self.lora_a),
                                      self.lora_b).to(x.dtype) * self.scale


def depths(count):
    return [math.ceil((i + 1) * count / 4) - 1 for i in range(4)]


def adapt_selected(module, blocks, targets, cfg):
    for index in depths(len(blocks)):
        block = blocks[index]
        for path, layer in list(block.named_modules()):
            if isinstance(layer, nn.Linear) and path.rsplit('.', 1)[-1] in targets:
                parent, _, name = path.rpartition('.')
                owner = block.get_submodule(parent) if parent else block
                setattr(owner, name, LoRALinear(layer, cfg['lora_rank'], cfg['lora_alpha']))
    module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})


def shared_frozen_copy(module):
    """Separate modules, buffers and trainable parameters; alias immutable base."""
    memo = {id(p): p for p in module.parameters() if not p.requires_grad}
    return copy.deepcopy(module, memo)


@contextmanager
def capture_depths(blocks):
    # Temporary hooks prevent deepcopy closures from retaining the online model.
    # non-reentrant checkpointing retains these readout tensors' gradient graph.
    values, handles = {}, []
    for j, i in enumerate(depths(len(blocks))):
        def capture(_module, _args, output, j=j):
            values[j] = output[0] if isinstance(output, tuple) else output
        handles.append(blocks[i].register_forward_hook(capture))
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


class StateEncoder(nn.Module):
    def __init__(self, language, vision, cfg):
        super().__init__()
        self.language, self.vision = language, vision
        hidden = language.get_input_embeddings().weight.shape[1]
        vwidth = vision.config.hidden_size
        adapt_selected(language, language.layers, LANGUAGE_TARGETS, cfg)
        adapt_selected(vision, vision.blocks, {'qkv', 'proj'}, cfg)
        self.jepa_adapter = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, hidden),
                                         nn.GELU(), nn.Linear(hidden, hidden))
        self.jepa_positions = nn.Parameter(torch.randn(1, cfg['visual_grid'] ** 2, hidden) * .02)
        self.slot_queries = nn.Parameter(torch.randn(8, WIDTH) * .02)
        self.language_query = nn.Linear(WIDTH, hidden)
        self.fusion_readouts = nn.ModuleList([nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, WIDTH))
                                             for _ in range(4)])
        self.visual_readouts = nn.ModuleList([nn.Sequential(nn.LayerNorm(vwidth), nn.Linear(vwidth, WIDTH))
                                             for _ in range(4)])
        self.visual_norm = nn.LayerNorm(WIDTH)

    def forward(self, features, text_ids, text_mask, image_inputs):
        # The two groups have distinct sources. Native visual slots never read text.
        with capture_depths(self.vision.blocks) as captured:
            self.vision(hidden_states=image_inputs['pixel_values'].to(next(self.vision.parameters()).dtype),
                        grid_thw=image_inputs['image_grid_thw'])
        batch = len(features)
        visual_slots = []
        grid = image_inputs['image_grid_thw']
        if len(grid) != batch or not torch.equal(grid, grid[:1].expand_as(grid)):
            raise ValueError('state readout requires one equally prepared square image per sample')
        for j in range(4):
            raw = captured[j].reshape(batch, -1, captured[j].shape[-1])
            values = self.visual_readouts[j](raw.float())
            query = self.slot_queries[4+j]
            weights = ((values * query).sum(-1) / math.sqrt(WIDTH)).softmax(-1)
            visual_slots.append(self.visual_norm((weights[..., None] * values).sum(1) + query))
        embed = self.language.get_input_embeddings()
        visual = (self.jepa_adapter(features.float()) + self.jepa_positions).to(embed.weight.dtype)
        query = self.language_query(self.slot_queries[:4])[None].expand(batch, -1, -1).to(embed.weight.dtype)
        seq = torch.cat([visual, embed(text_ids), query], 1)
        mask = torch.cat([text_mask.new_ones(visual.shape[:2]), text_mask, text_mask.new_ones(query.shape[:2])], 1)
        positions = (mask.cumsum(-1) - 1).clamp_min(0)
        with capture_depths(self.language.layers) as captured:
            self.language(inputs_embeds=seq, attention_mask=mask, position_ids=positions,
                          use_cache=False, return_dict=True)
        fusion = [self.fusion_readouts[j](captured[j][:, -4+j].float()) for j in range(4)]
        return torch.stack(fusion + visual_slots, 1)


class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        width = cfg['lwm_width']
        self.input = nn.Linear(WIDTH, width)
        self.horizon = nn.Embedding(4, width)
        self.positions = nn.Parameter(torch.randn(1, 8, width) * .02)
        layer = nn.TransformerEncoderLayer(width, 8, width*4, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, cfg['lwm_depth'], enable_nested_tensor=False)
        self.output = nn.Linear(width, WIDTH)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state, horizon):
        state = F.layer_norm(state.float(), (WIDTH,))
        values = self.input(state) + self.positions + self.horizon(horizon)[:, None]
        return state + self.output(self.transformer(values))


class NativeForecast(nn.Module):
    def __init__(self, cfg, condition='slots', device='cuda'):
        super().__init__()
        if condition not in ('native', 'slots', 'shuffled'):
            raise ValueError(condition)
        self.cfg, self.condition = dict(cfg), condition
        self.decoder = load_qwen(cfg)
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(cfg['qwen'], local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = 'left'
        hidden = self.decoder.config.text_config.hidden_size
        # All conditions construct their decoder and finding head before auxiliaries.
        self.finding = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, len(cfg['findings'])))
        if condition != 'native':
            language = shared_frozen_copy(self.decoder.model.language_model)
            vision = shared_frozen_copy(self.decoder.model.visual)
        adapt_selected(self.decoder.model.language_model, self.decoder.model.language_model.layers, LANGUAGE_TARGETS, cfg)
        adapt_selected(self.decoder.model.visual, self.decoder.model.visual.blocks, {'qkv', 'proj'}, cfg)
        self.encoder = None
        self.target_encoder = None
        if condition != 'native':
            self.encoder = StateEncoder(language, vision, cfg)
            self.world = WorldModel(cfg)
            self.slot_projection = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, hidden))
            self.state_finding = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, len(cfg['findings'])))
        self.metadata = dict(architecture='native_qwen_with_optional_predicted_4plus4_slots_v1',
            condition=condition, qwen=cfg['qwen'], native_image_pixels=cfg.get('native_image_pixels', 512),
            decoder_inputs=['current_raw_image', 'current_report_and_prior_ehr', 'requested_horizon'] +
                           ([] if condition == 'native' else ['predicted_future_slots']),
            state_shape=[8, WIDTH], native_vision_retained=True,
            frozen_jepa_cache='auxiliary fusion state only', target_update='fixed Stage1 snapshot',
            finding_probability='same supervised BCE head on decoder source prompt',
            decoder_lora_depths=depths(len(self.decoder.model.language_model.layers)),
            visual_lora_depths=depths(len(self.decoder.model.visual.blocks)),
            stage1_tasks=['current_native_report', 'state_only_report', 'state_finding'],
            complete_six_task_stage1=False, future_observation_in_decoder=False,
            shared_immutable_backbone=True, independent_encoder_decoder_adapters=True)
        self.to(device)

    @property
    def device(self):
        return next(self.decoder.parameters()).device

    def train(self, mode=True):
        super().train(mode)
        if self.target_encoder is not None:
            self.target_encoder.eval()
        return self

    def begin_stage2(self):
        if self.encoder is not None and self.target_encoder is None:
            self.target_encoder = shared_frozen_copy(self.encoder).requires_grad_(False).eval()
            self.target_encoder.language.gradient_checkpointing_disable()
            self.target_encoder.vision.gradient_checkpointing_disable()

    def image_inputs(self, images):
        pixels = self.cfg.get('native_image_pixels', 512)
        inputs = self.processor.image_processor(images=images, return_tensors='pt',
                                                min_pixels=pixels*pixels, max_pixels=pixels*pixels)
        return {k: v.to(self.device) for k, v in inputs.items() if torch.is_tensor(v)}

    def state(self, batch, side='source', image_only=False, target=False):
        encoder = self.target_encoder if target else self.encoder
        if encoder is None:
            raise RuntimeError('State encoder not initialized')
        image_key = 'donor_images' if side == 'donor' else '_'+side+'_images'
        ids, mask = batch[side+'_ids'], batch[side+'_mask']
        if image_only:
            ids, mask = ids[:, :0], mask[:, :0]
        return encoder(batch[side+'_features'], ids, mask, self.image_inputs(batch[image_key]))

    def source_inputs(self, batch, stage=2):
        if stage == 1:
            texts = ['Describe this chest radiograph. Write only FINDINGS and IMPRESSION, without reasoning or introductory text.'] * len(batch['_source_images'])
        else:
            context = self.tokenizer.batch_decode(batch['source_ids'], skip_special_tokens=True)
            texts = [text+'\n\nRequested follow-up horizon: '+HORIZONS[int(h)]+'. '
                     'Only the supplied current study is the comparison baseline. No future study has been observed.\n'
                     'Predict the follow-up chest radiograph report. Write only FINDINGS and IMPRESSION, without reasoning or introductory text.'
                     for text, h in zip(context, batch['horizon'])]
        conversations = [[{'role': 'user', 'content': [{'type': 'image', 'image': image},
                           {'type': 'text', 'text': text}]}] for image, text in zip(batch['_source_images'], texts)]
        pixels = self.cfg.get('native_image_pixels', 512)
        values = self.processor.apply_chat_template(conversations, tokenize=True, return_dict=True,
            return_tensors='pt', add_generation_prompt=True, enable_thinking=False,
            processor_kwargs={'text_kwargs': {'padding': True},
                              'images_kwargs': {'min_pixels': pixels*pixels, 'max_pixels': pixels*pixels}})
        return {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in values.items()}

    def native_prefix(self, inputs, slots=None):
        """Preserve Qwen image embeddings and M-RoPE, add soft tokens in user turn."""
        model = self.decoder.model
        ids, mask, types = inputs['input_ids'], inputs['attention_mask'], inputs['mm_token_type_ids']
        embeds = model.get_input_embeddings()(ids)
        image_output = model.get_image_features(inputs['pixel_values'], inputs['image_grid_thw'], return_dict=True)
        image_embeds = torch.cat(image_output.pooler_output, dim=0).to(embeds.dtype)
        image_mask, _ = model.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=image_embeds)
        embeds = embeds.masked_scatter(image_mask, image_embeds)
        if slots is not None:
            values = self.slot_projection(slots.float()).to(embeds.dtype)
            end_token = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
            ends = [torch.where(row == end_token)[0][-1].item() for row in ids]
            if len(set(ends)) != 1:
                raise ValueError('left-padded native prompt must end user turn at same batch position')
            cut, count = ends[0], values.shape[1]
            ids = torch.cat([ids[:, :cut], ids.new_full((len(ids), count), self.tokenizer.pad_token_id), ids[:, cut:]], 1)
            mask = torch.cat([mask[:, :cut], mask.new_ones((len(mask), count)), mask[:, cut:]], 1)
            types = torch.cat([types[:, :cut], types.new_zeros((len(types), count)), types[:, cut:]], 1)
            embeds = torch.cat([embeds[:, :cut], values, embeds[:, cut:]], 1)
        positions, delta = model.get_rope_index(ids, mm_token_type_ids=types,
                            image_grid_thw=inputs['image_grid_thw'], attention_mask=mask)
        return dict(embeds=embeds, mask=mask, positions=positions, delta=delta)

    def state_only_prefix(self, state):
        ids = self.tokenizer.apply_chat_template([{'role':'user', 'content':
            'Describe the supplied chest radiograph state. Write only FINDINGS and IMPRESSION.'}],
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=False)
        embed = self.decoder.model.get_input_embeddings()
        text = embed(torch.tensor(ids, device=self.device))[None].expand(len(state), -1, -1)
        values = self.slot_projection(state.float()).to(text.dtype)
        seq = torch.cat([values, text], 1)
        mask = torch.ones(seq.shape[:2], device=self.device, dtype=torch.long)
        positions = torch.arange(seq.shape[1], device=self.device)[None].expand(len(state), -1)
        return dict(embeds=seq, mask=mask, positions=positions[None].expand(3, -1, -1),
                    delta=mask.new_zeros((len(state), 1)))

    def decode_loss(self, prefix, targets, target_mask):
        length = prefix['embeds'].shape[1]
        text = self.decoder.model.get_input_embeddings()(targets)
        mask = torch.cat([prefix['mask'], target_mask], 1)
        offset = prefix['positions'].amax(dim=(0, 2))[:, None] + 1
        continuation = offset + torch.arange(targets.shape[1], device=self.device)[None]
        positions = torch.cat([prefix['positions'], continuation[None].expand(3, -1, -1)], -1)
        output = self.decoder.model.language_model(inputs_embeds=torch.cat([prefix['embeds'], text], 1),
                    attention_mask=mask, position_ids=positions, use_cache=False, return_dict=True).last_hidden_state
        ce = chunked_ce(output[:, length-1:-1], targets.masked_fill(~target_mask.bool(), -100),
                        self.decoder.lm_head.weight, chunk=self.cfg.get('ce_chunk', 32))
        return ce, self.finding(output[:, length-1].float())

    def losses(self, batch, stage=2):
        if stage == 1:
            if self.encoder is None or self.condition != 'slots':
                raise ValueError('train one shared slots Stage1, then initialize all Stage2 branches')
            state = self.state(batch, image_only=True)
            state_ce, _ = self.decode_loss(self.state_only_prefix(state), batch['source_target_ids'], batch['source_target_mask'])
            native_ce, scores = self.decode_loss(self.native_prefix(self.source_inputs(batch, stage=1)),
                                               batch['source_target_ids'], batch['source_target_mask'])
            bce = masked_bce(self.state_finding(state.float().mean(1)), batch['source_labels'])
            native_bce = masked_bce(scores, batch['source_labels'])
            total = native_ce + state_ce + self.cfg['finding_weight'] * (bce + native_bce)
            return total, dict(text=native_ce, state_text=state_ce, finding=bce, native_finding=native_bce)
        state = None
        latent = torch.zeros((), device=self.device)
        if self.encoder is not None:
            # Shuffle only auxiliary current evidence. Decoder keeps the patient's source.
            side = 'donor' if self.condition == 'shuffled' else 'source'
            state = self.world(self.state(batch, side), batch['horizon'])
            with torch.no_grad():
                target = self.state(batch, 'target', target=True)
            latent = F.mse_loss(F.layer_norm(state.float(), (WIDTH,)), F.layer_norm(target.float(), (WIDTH,)))
        ce, scores = self.decode_loss(self.native_prefix(self.source_inputs(batch), state),
                                     batch['target_target_ids'], batch['target_target_mask'])
        bce = masked_bce(scores, batch['target_labels'])
        total = self.cfg.get('text_weight', 1.)*ce + self.cfg['finding_weight']*bce + self.cfg['latent_weight']*latent
        return total, dict(text=ce, finding=bce, latent=latent)

    @torch.no_grad()
    def predict(self, batch):
        state = None if self.encoder is None else self.world(
            self.state(batch, 'donor' if self.condition == 'shuffled' else 'source'), batch['horizon'])
        prefix = self.native_prefix(self.source_inputs(batch), state)
        language = self.decoder.model.language_model
        out = language(inputs_embeds=prefix['embeds'], attention_mask=prefix['mask'],
                       position_ids=prefix['positions'], use_cache=True, return_dict=True)
        scores = self.finding(out.last_hidden_state[:, -1].float()).sigmoid()
        mask, cache = prefix['mask'], out.past_key_values
        token = F.linear(out.last_hidden_state[:, -1].to(self.decoder.lm_head.weight.dtype), self.decoder.lm_head.weight).argmax(-1)
        generated, ended = [], torch.zeros(len(token), dtype=torch.bool, device=self.device)
        for index in range(self.cfg['generation_tokens']):
            generated.append(torch.where(ended, self.tokenizer.pad_token_id, token))
            ended |= token == self.tokenizer.eos_token_id
            if ended.all() or index + 1 == self.cfg['generation_tokens']:
                break
            mask = torch.cat([mask, mask.new_ones((len(mask), 1))], 1)
            # Same absolute multimodal offsets as native Qwen cached generation.
            position = mask.sum(-1, keepdim=True) - 1 + prefix['delta']
            out = language(input_ids=generated[-1][:, None], attention_mask=mask,
                           position_ids=position[None].expand(3, -1, -1), past_key_values=cache,
                           use_cache=True, return_dict=True)
            cache = out.past_key_values
            token = F.linear(out.last_hidden_state[:, -1].to(self.decoder.lm_head.weight.dtype), self.decoder.lm_head.weight).argmax(-1)
        reports = self.tokenizer.batch_decode(torch.stack(generated, 1), skip_special_tokens=True)
        return reports, scores

    @staticmethod
    def compact_key(key):
        if key.startswith('decoder.'):
            return '.lora_' in key
        if key.startswith(('encoder.', 'target_encoder.')) and ('.language.' in key or '.vision.' in key):
            return '.lora_' in key
        return True

    def compact_state(self):
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items() if self.compact_key(k)}

    def load_compact(self, state):
        has_target = any(k.startswith('target_encoder.') for k in state)
        if has_target and self.encoder is not None:
            self.begin_stage2()
        if self.encoder is None:
            state = {k: v for k, v in state.items() if k.startswith(('decoder.', 'finding.'))}
        missing, unexpected = self.load_state_dict(state, strict=False)
        required = [k for k in missing if self.compact_key(k) and
                    (has_target or not k.startswith('target_encoder.'))]
        if required or unexpected:
            raise ValueError(dict(missing=required, unexpected=unexpected))

    def gradient_audit(self, stage):
        totals = {}
        for name, parameter in self.named_parameters():
            if parameter.grad is None:
                continue
            if not torch.isfinite(parameter.grad).all():
                raise ValueError('non-finite gradient: '+name)
            if name.startswith('target_encoder.') or '.base.' in name:
                raise ValueError('gradient reached frozen parameter: '+name)
            group = ('.'.join(name.split('.')[:3]) if name.startswith('decoder.') else
                     '.'.join(name.split('.')[:2]) if name.startswith('encoder.') else name.split('.')[0])
            totals[group] = totals.get(group, 0.) + float(parameter.grad.float().square().sum())
        result = {k: math.sqrt(v) for k, v in totals.items()}
        if not any(k.startswith('decoder.model.language') and v > 0 for k, v in result.items()):
            raise ValueError('decoder language gradient missing')
        if not result.get('decoder.model.visual', 0) > 0:
            raise ValueError('native decoder vision gradient missing')
        if self.encoder is not None:
            for key in ('encoder.language', 'encoder.vision', 'encoder.jepa_adapter',
                        'encoder.slot_queries', 'slot_projection'):
                if result.get(key, 0) <= 0:
                    raise ValueError('state branch gradient missing: '+key)
            if stage == 2 and result.get('world', 0) <= 0:
                raise ValueError('predictor gradient missing')
        return result
