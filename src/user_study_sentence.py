import random
from datetime import datetime
import argparse
import os
import PIL.Image
import jieba
import re
import pandas as pd
from psychopy import core, visual, event

from utilities import *
from trigger_test import send_trigger


def main_study_sentence(win, stu_list, fw, n=20):
    split_stu_list = split_arr(stu_list, n)
    fw.write("-----------------main_study-----------------\n")
    for idx, stu_list in enumerate(split_stu_list):
        if idx != 0:
            if idx == 3:
                instruction_rest = visual.TextStim(win,
                                                   text='阶段' + str(idx) + '休息,\n第一部分结束\n按下空格继续实验',
                                                   pos=(0, TEXT_HEIGHT), height=WORD_HEIGHT, color=(1, 1, 1),
                                                   alignText='center')
            else:
                instruction_rest = visual.TextStim(win, text='阶段' + str(idx) + '休息,\n按下空格继续实验',
                                                   pos=(0, TEXT_HEIGHT), height=WORD_HEIGHT, color=(1, 1, 1),
                                                   alignText='center')
            instruction_rest.draw()
            win.flip()

            event.waitKeys(keyList=['space'])
            core.wait(0.5)
        for stu in stu_list:
            # 1
            text_before_img = ["请仔细观察图片"]
            text_ui = visual.TextStim(win, text=split_text(text_before_img), pos=(-TEXT_LEFT, TEXT_HEIGHT),
                                      height=WORD_HEIGHT, color=(1, 1, 1), alignText='center')
            text_ui.draw()
            win.flip()
            core.wait(1)  # 等待1s
            # 2
            img = PIL.Image.open(stu['image_path'])
            scale = min(MIN_SIZE[0] / img.size[0], MIN_SIZE[1] / img.size[1])
            image_ui = visual.ImageStim(win, image=stu['image_path'], pos=(0, IMAGE_BIAS_HEIGHT_1),
                                        size=(img.size[0] * scale, img.size[1] * scale))
            image_ui.draw()
            win.flip()
            core.wait(0.010)  # 延迟10ms
            send_trigger(TRIGGER_LIST['image'])
            core.wait(6)  # 展示图片6s
            # 3
            text_ui = visual.TextStim(win, text='+', pos=(0, IMAGE_BIAS_HEIGHT_1), height=WORD_HEIGHT, color=(1, 1, 1),
                                      alignText='center')
            text_ui.draw()
            win.flip()
            core.wait(0.010)  # 延迟10ms
            send_trigger(TRIGGER_LIST['+'])
            core.wait(1)
            # 4
            for word in stu["text"]:
                text_ui = visual.TextStim(win, text=word, pos=(0, IMAGE_BIAS_HEIGHT_1), height=WORD_HEIGHT,
                                          color=(1, 1, 1), alignText='center')
                text_ui.draw()
                win.flip()
                core.wait(0.010)  # 延迟10ms
                send_trigger(TRIGGER_LIST['text'])
                core.wait(0.75)
            # 5
            text_ui = visual.TextStim(win,
                                      text=split_text(["以上句子是否出现幻觉", "输入 Y 表示有", "输入 N 表示没有"]),
                                      pos=(0, TEXT_HEIGHT), height=WORD_HEIGHT, color=(1, 1, 1), alignText='center')
            text_ui.draw()
            win.flip()
            core.wait(0.010)  # 延迟10ms
            send_trigger(TRIGGER_LIST['judgement'])
            keys = event.waitKeys(keyList=['y', 'n'])
            stu["judgement"] = keys[0]
            fw.write(str(stu) + '\n')


def user_study_main(stu_list, fw):
    fw.write("-----------------stu_list-----------------\n")
    fw.write(str(stu_list) + '\n' + '\n')

    win = visual.Window(size=MIN_SIZE, fullscr=False, screen=1, winType='pyglet', allowGUI=False, allowStencil=False,
                        monitor='testMonitor', color=[-1, -1, -1], colorSpace='rgb', blendMode='avg', useFBO=True,
                        units='pix')

    instruction_end = visual.TextStim(win, text='实验结束,请联系主试', pos=(0, TEXT_HEIGHT), height=WORD_HEIGHT,
                                      color=(1, 1, 1),
                                      alignText='center')
    instruction_main = visual.TextStim(win, text='按空格键,开始正式实验', pos=(0, TEXT_HEIGHT), height=WORD_HEIGHT,
                                       color=(1, 1, 1), alignText='center')

    instruction_main.draw()
    win.flip()
    event.waitKeys(keyList=['space'])

    main_study_sentence(win, stu_list, fw)

    instruction_end.draw()
    win.flip()
    event.waitKeys(keyList=['space'])
    win.close()
    core.quit()
    fw.close()


def load_question_info(annotation_file):
    df = pd.read_csv(annotation_file)
    _total_arr = []
    for idx, row in df.iterrows():
        words = jieba.lcut(row["text_ch"])  # 返回一个列表
        punctuation_pattern = r'[\s+\.\!\/_,$%^*()<>《》+\"\']+|[——！，。？、~@#￥%……&*（）【】“”‘’：；「」·]'
        filtered_words = [word for word in words if not re.match(punctuation_pattern, word)]
        _total_arr.append(
            {"id": row["id"], "image_path": f"../data/AMBER_image/AMBER_{row['id']}.jpg", "text": filtered_words,
             "type": row["type"], "hallu_word": row["hallu_word"]})
    pairs = [(_total_arr[2 * k], _total_arr[2 * k + 1]) for k in range(len(_total_arr) // 2)]
    first, second = zip(*[random.sample(pair, 2) for pair in pairs])
    first = list(first)
    second = list(second)
    random.shuffle(first)
    random.shuffle(second)
    return first + second


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Collect EEG data while reading LLM hallucination content.')
    parser.add_argument('-s', '--seed', type=int, default=54321, help='random seed')
    args = parser.parse_args()
    random.seed(args.seed)

    total_arr = load_question_info('../data/annotation_sentence.csv')
    user_study_main(total_arr,
                    open(os.path.join('../record/', f"{datetime.now().strftime('%Y%m%d')}_{args.seed}_sentence.txt"),
                         'w', encoding='utf8'))
