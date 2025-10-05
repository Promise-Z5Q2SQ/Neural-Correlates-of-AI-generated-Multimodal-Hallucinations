HEIGHT = 1500
WIDE = 1500
MIN_SIZE = [WIDE, HEIGHT]

TEXT_LEFT = WIDE / 10  # 文字向左偏移
TEXT_HEIGHT = HEIGHT / 10  # 文字向上偏移
WORD_HEIGHT = WIDE / 20  # 文字高度

IMAGE_GAP = WIDE / 25  # 图片间的距离
IMAGE_SIZE_1 = (WIDE / 1.9, WIDE / 1.9)  # 图片大小
IMAGE_SIZE_2 = (WIDE / 3, WIDE / 3)
IMAGE_BIAS_HEIGHT_1 = HEIGHT / 15  # 图片向上偏移
IMAGE_BIAS_HEIGHT_2 = HEIGHT / 8
TEXT_POS_LIST_2 = [  # 2张图时文字的位置
    [-IMAGE_SIZE_2[0] / 2 - IMAGE_GAP / 2, -IMAGE_SIZE_2[1] / 2],
    [IMAGE_SIZE_2[0] / 2 + IMAGE_GAP / 2, -IMAGE_SIZE_2[1] / 2]
]
POS_LIST_2 = [  # 2张图时图片的位置
    [-IMAGE_SIZE_2[0] / 2 - IMAGE_GAP / 2, IMAGE_BIAS_HEIGHT_2],
    [IMAGE_SIZE_2[0] / 2 + IMAGE_GAP / 2, IMAGE_BIAS_HEIGHT_2]
]

TRIGGER_LIST = {
    'image': 100,
    '+': 105,  # 展示‘+’
    'text': 116,
    'judgement': 322,
}

def split_arr(stu_list, n):
    split_list = []
    for i in range(0, len(stu_list), n):
        split_list.append(stu_list[i: i + n])
    return split_list


def split_str(input_text):
    output_text = ''
    line_length = 15
    while len(input_text) > line_length:
        output_text += input_text[:line_length] + '\n'
        input_text = input_text[line_length:]
    output_text += input_text
    return output_text


def split_text(input_text):
    if type(input_text) == str:
        return split_str(input_text)
    output_text = ''
    for idx, input_text_line in enumerate(input_text):
        output_text += split_str(input_text_line)
        if idx != len(input_text) - 1:
            output_text += '\n\n'
    return output_text
