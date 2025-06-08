import os
from collections import Counter
import argparse
import os
from collections import Counter
import json
import numpy as np
import random
from data.Amazon.parse import AmazonDataprocess
from data.Children.parse import ChildrenDataprocess
from data.Finance.parse import FinanceDataprocess
from data.Math.parse import MathDataprocess


def setup_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def print_args(args):
    print("="*40,flush=True)
    print("\nParsed arguments:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("="*40,flush=True)
    print()

def save_dict(word_counts, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(word_counts, f, ensure_ascii=False, indent=4)

def count_words(text):
    words = text.split()
    return dict(Counter(words))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Sketch Project")
    parser.add_argument('--datasetName', type=str, default="Children", help='nameOfDataset.')
    parser.add_argument('--Num_In', type=int, default=800000)
    parser.add_argument('--SampleRate', type=float, default=1e-5)
    parser.add_argument('--SaveName',type=str,default="Children_Sample1e-5")
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    setup_seed(args.seed)
    print_args(args)
    if args.datasetName =='Children':
        stringList = ChildrenDataprocess(args.Num_In)
    elif args.datasetName == 'Finance':
        stringList = FinanceDataprocess(args.Num_In)
    elif args.datasetName == 'Amazon':
        stringList = AmazonDataprocess(args.Num_In)
    elif args.datasetName == 'Math':
        stringList = MathDataprocess(args.Num_In)
    
    content = "".join(stringList)
    wordNum = len(content.split())
    print(wordNum)
    wordList =  content.split()
    ndv= len(set(wordList))

    if args.SampleRate == 1:
        sampleList = wordList
        content_sample = content
    else:      
        sampleList = random.sample(wordList, int(args.SampleRate*wordNum))
        content_sample = " ".join(sampleList)
    realSampleRatio = (len(content_sample.split())) / (wordNum) 
    print(f"Sample {int(args.SampleRate*wordNum)} from {wordNum}")
    wordCnt = count_words(content_sample)

    if len(wordCnt) > 10000:
        # Sort words by frequency in descending order and take top 10,000
        top_words = sorted(wordCnt.items(), key=lambda x: x[1], reverse=True)[:10000]
        wordCnt = dict(top_words)

    wordCnt['<total>'] = wordNum
    wordCnt['<numwords>'] = int(args.SampleRate * wordNum)

    wordCnt['<sel>'] = realSampleRatio

    save_dict(wordCnt,'./cache/' + args.SaveName)
    print("Save Done!")
