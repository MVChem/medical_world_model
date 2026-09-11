"""Conservative review routing, separate from unchanged raw model output."""
VERSION='agreement-review-routing-v1'


def triage(label,flags):
    reasons=set(flags)
    primary=label['primary_class']
    if primary=='indeterminate': reasons.add('model_indeterminate')
    if label['image_assessment']!=primary: reasons.add('image_does_not_support_primary_class')
    if label['report_assessment']!=primary: reasons.add('report_does_not_support_primary_class')
    if primary=='changed' and label['direction']=='uncertain': reasons.add('uncertain_change_direction')
    return {'screening_class':'needs_review' if reasons else primary,'reasons':sorted(reasons)}
