true_mask = imread('./inputs/oval_and_rectangle_true_mask.png');
exp_mask = imread('./Outputs/or_seg2_mask.png');

int_mask = true_mask & exp_mask;
int_card = sum(int_mask(:));
union_mask = true_mask | exp_mask;
union_card = sum(union_mask(:));

true_card = sum(true_mask(:));
exp_card = sum(exp_mask(:));

dsc = (2*int_card)./(true_card + exp_card)
IOU = int_card ./ union_card