import json
from pprint import pprint

from lhotse import load_manifest_lazy

from jiwer import wer

def calc_wer():

    dev_cuts = load_manifest_lazy('/export/c01/ashah108/vad/data/tedlium/manifests/tedlium_no_tedlium_cuts_dev_1_fbank.jsonl.gz')
    test_cuts = load_manifest_lazy('/export/c01/ashah108/vad/data/tedlium/manifests/tedlium_no_tedlium_cuts_test_1_fbank.jsonl.gz')

    dev_cuts_no_split = load_manifest_lazy('/export/c01/ashah108/vad/data/predictions/dev_cuts_tedlium_no_tedlium_buffer-1_no_split.jsonl.gz')
    test_cuts_no_split = load_manifest_lazy('/export/c01/ashah108/vad/data/predictions/test_cuts_tedlium_no_tedlium_buffer-1_no_split.jsonl.gz')

    dev_cuts_trim = load_manifest_lazy('/export/c01/ashah108/vad/data/tedlium/manifests/trim_tedlium_cuts_dev_fbank.jsonl.gz')
    test_cuts_trim = load_manifest_lazy('/export/c01/ashah108/vad/data/tedlium/manifests/trim_tedlium_cuts_test_fbank.jsonl.gz')

    dev_hyp = json.load(open('/export/c01/ashah108/vad/data/wer/tedlium_no_tedlium_dev-1.json'))
    test_hyp = json.load(open('/export/c01/ashah108/vad/data/wer/tedlium_no_tedlium_test-1.json'))

    dev_pred, test_pred, dev_gt, test_gt = [], [], [], []

    for cut in dev_cuts:
        cut_id = cut.id
        if cut_id in dev_hyp:
            dev_pred += dev_hyp[cut_id]
    
    for cut in test_cuts:
        cut_id = cut.id
        if cut_id in test_hyp:
            test_pred += test_hyp[cut_id]

    for cut in dev_cuts_trim:
        text = cut.supervisions[0].text
        dev_gt += text.split()

    for cut in test_cuts_trim:
        text = cut.supervisions[0].text
        test_gt += text.split()


    dev_error = wer(' '.join(dev_gt), ' '.join(dev_pred))
    test_error = wer(' '.join(test_gt), ' '.join(test_pred))

    print(f'Dev WER: {100 * round(dev_error, 4)}')
    print(f'Test WER: {100 * round(test_error, 4)}')

    # save all these to different files
    with open('/export/c01/ashah108/vad/data/wer/dev_pred-tedlium_no_tedlium.txt', 'w') as f:
        f.write(str(dev_pred))

    with open('/export/c01/ashah108/vad/data/wer/test_pred-tedlium_no_tedlium.txt', 'w') as f:
        f.write(str(test_pred))

    with open('/export/c01/ashah108/vad/data/wer/dev_gt-tedlium_no_tedlium.txt', 'w') as f:
        f.write(str(dev_gt))
    
    with open('/export/c01/ashah108/vad/data/wer/test_gt-tedlium_no_tedlium.txt', 'w') as f:
        f.write(str(test_gt))


    def calc_common_words(curr_cuts, curr_hyp, curr_cuts_no_split, phase='dev'):
        merged_cuts_ids, temp = [], []
        for i, cut in enumerate(curr_cuts):
            if i == 0:
                temp.append(cut)
            elif round(cut.start, 2) == round(temp[-1].start + temp[-1].duration, 2):
                temp.append(cut)
            else:
                if len(temp) > 0:
                    merged_cuts_ids.append(temp)
                temp = [cut]
        
        if len(temp) > 0:
            merged_cuts_ids.append(temp)

        
        pred_texts, temp_texts = [], []
        for cuts in merged_cuts_ids:
            for cut in cuts:
                if cut.id in curr_hyp:
                    temp_texts += curr_hyp[cut.id]
            pred_texts.append(' '.join(temp_texts))
            temp_texts = []

        gt_texts = []
        for cut in curr_cuts_no_split:
            gt_texts.append(cut.supervisions[0].text)


        x, y = 0, 0
        for gt, pred in zip(gt_texts, pred_texts):
            x += wer(gt, pred)

            gt_words = gt.split()
            pred_words = pred.split()

            common_words = 0
            common_words_arr = []
            # s1: b c d e f g a l m
            # s2: a b x y d z g

            i, j = len(gt_words) - 1, len(pred_words) - 1
            index = len(gt_words)
            while i >= 0 and j >= 0:
                if pred_words[j] in gt_words[:index]:
                    index = len(gt_words) - gt_words[:index][::-1].index(pred_words[j])
                    common_words += 1
                    common_words_arr.append(pred_words[j])
                    i = index - 1
                    j -= 1
                else:
                    j -= 1

            den = len(pred_words) if len(pred_words) > 0 else 1
            y += common_words / den
            print(f'{phase}: % common words: ', 100 * round(common_words / den, 2))


            # print(gt)
            # print(pred)
            # print('common words: ', ' '.join(common_words_arr[::-1]))
            # print(wer(gt, pred))
        

        print(f'{phase}: avg wer per cut: ', 100 * round(x / len(pred_texts), 4))
        print(f'{phase}: avg % common words per cut: ', 100 * round(y / len(pred_texts), 2))
        # print('overall wer: ', wer(' '.join(gt_texts), ' '.join(pred_texts)))    


    calc_common_words(dev_cuts, dev_hyp, dev_cuts_no_split, phase='dev')
    calc_common_words(test_cuts, test_hyp, test_cuts_no_split, phase='test')

    