"""Native Qwen3.5 vision-language baseline: current image/report/bin -> future report."""
import torch
from torch import nn
from PIL import Image, ImageOps
from transformers import AutoProcessor

from model import load_qwen, adapt, masked_bce, chunked_ce, HORIZON_NAMES


class DirectQwen(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.qwen = load_qwen(cfg)
        self.qwen.model.language_model = adapt(self.qwen.model.language_model, cfg)
        self.qwen.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        self.processor = AutoProcessor.from_pretrained(cfg['qwen'], local_files_only=True)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = 'left'
        self.finding = nn.Sequential(nn.LayerNorm(self.qwen.config.text_config.hidden_size),
                                     nn.Linear(self.qwen.config.text_config.hidden_size, len(cfg['findings'])))

    def source_inputs(self, batch):
        conversations = []
        texts = self.tokenizer.batch_decode(batch['source_ids'], skip_special_tokens=True)
        for path, report, horizon in zip(batch['_source_images'], texts, batch['horizon'].tolist()):
            with Image.open(path) as img:
                image = ImageOps.pad(img.convert('RGB'), (256, 256), method=Image.Resampling.BICUBIC)
            conversations.append([{'role': 'user', 'content': [
                {'type': 'image', 'image': image},
                {'type': 'text', 'text': f'Current chest radiograph and available current evidence:\n{report}\nPredict the follow-up chest radiograph report at {HORIZON_NAMES[horizon]}. Write FINDINGS and IMPRESSION.'}
            ]}])
        values = self.processor.apply_chat_template(conversations, tokenize=True, return_dict=True,
            return_tensors='pt', add_generation_prompt=True, enable_thinking=False,
            processor_kwargs={'text_kwargs': {'padding': True}, 'images_kwargs': {'min_pixels': 256*256, 'max_pixels': 256*256}})
        return {k: v.to('cuda') if torch.is_tensor(v) else v for k, v in values.items()}

    def losses(self, batch, stage=2):
        inputs = self.source_inputs(batch)
        length = inputs['input_ids'].shape[1]
        targets, mask = batch['target_target_ids'], batch['target_target_mask']
        inputs['input_ids'] = torch.cat([inputs['input_ids'], targets], 1)
        inputs['attention_mask'] = torch.cat([inputs['attention_mask'], mask], 1)
        if 'mm_token_type_ids' in inputs:
            inputs['mm_token_type_ids'] = torch.cat([inputs['mm_token_type_ids'], torch.zeros_like(targets)], 1)
        outputs = self.qwen.model(**inputs, use_cache=False).last_hidden_state
        ce = chunked_ce(outputs[:, length-1:-1], targets.masked_fill(~mask.bool(), -100), self.qwen.lm_head.weight)
        # The final source prompt position cannot attend to the teacher-forced target.
        scores = self.finding(outputs[:, length-1].float())
        bce = masked_bce(scores, batch['target_labels'])
        return ce + self.cfg['finding_weight']*bce, dict(text=ce, finding=bce)

    @torch.no_grad()
    def predict(self, batch):
        inputs = self.source_inputs(batch)
        output = self.qwen.model(**inputs, use_cache=False).last_hidden_state
        scores = self.finding(output[:, -1].float()).sigmoid()
        tokens = self.qwen.generate(**inputs, max_new_tokens=self.cfg['generation_tokens'], do_sample=False,
                                    use_cache=True, eos_token_id=self.tokenizer.eos_token_id,
                                    pad_token_id=self.tokenizer.pad_token_id)
        return self.tokenizer.batch_decode(tokens[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True), scores

    def compact_state(self):
        return {k: v.detach().cpu() for k, v in self.state_dict().items() if 'lora_' in k or k.startswith('finding.')}

    def load_compact(self, state):
        missing, unexpected = self.load_state_dict(state, strict=False)
        if unexpected or any('lora_' in k or k.startswith('finding.') for k in missing):
            raise ValueError((missing, unexpected))
