import torch
from model import CaptchaCNN
from dataset import get_eval_data_loader
from encoding import decode_predict, decode

def main():
    model = CaptchaCNN()
    model.load_state_dict(torch.load('model/best_model.pkl'))
    model.eval()

    loader = get_eval_data_loader(batch_size=1)
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            outputs = model(images)
            outputs = outputs.squeeze(0)

            pred_text = decode_predict(outputs)
            true_text = decode(labels.squeeze(0))

            if pred_text == true_text:
                correct += 1
            total += 1

            if total % 200 == 0:
                print(f'已评估 {total} 张，当前准确率: {correct/total:.4f}')

    print(f'评估完成！总准确率: {correct/total:.4f} ({correct}/{total})')

if __name__ == '__main__':
    main()
