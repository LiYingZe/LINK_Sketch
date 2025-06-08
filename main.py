import os
from collections import Counter
import re
import argparse
import os
import random
from openai import OpenAI
import heapq
import re
from collections import Counter
import json
import numpy as np
import random


cache_dir = './cache'
def load_dict_from_json(filename):
    filepath = os.path.join(cache_dir, filename)
    try:
        with open(filepath, 'r') as f:
            dictionary = json.load(f)
        print(f"loaded {filepath}")
        return dictionary
    except FileNotFoundError:
        print(f"Error: file {filepath} not exist.")
        return None
    except json.JSONDecodeError:
        print(f"Error: file {filepath} is not valid JSON.")
        return None

def calculate_recall(predicted_words, actual_top_words):
    
    if not actual_top_words:
        return 0.0
    
    predicted_set = set(word.lower() for word in predicted_words)
    actual_set = set(word.lower() for word, _ in actual_top_words)
    
    correct = len(predicted_set & actual_set)
    return correct / len(actual_set)

def predict_single_Sil(input_text: str,model:str) -> tuple:
    
    client = OpenAI(api_key="sk-fxpgexcxnzlntgsjsxeaaaeonbjkcsihsazypaccffngifdq", 
                    base_url="https://api.siliconflow.cn/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {'role': 'user', 
            'content': input_text}
        ],
        stream=False,
    )
    return response.choices[0].message.content

def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    return text

def get_top_k_words(content, k=10):
    
    words = preprocess_text(content).split()
    word_counts = Counter(words)
    top_k = word_counts.most_common(k)
    return [word for word, count in top_k]

def parse_model_output(output_str):
    # Parse JSON string to Python dictionary
    result = None
    try:
        result = json.loads(output_str)
        
        # # Verify required keys
        assert "top_elements" in result
        
        # Verify frequency entries are numeric
        for v in result["top_elements"].values():
            assert isinstance(v, (int, float))
            
    except (json.JSONDecodeError, AssertionError) as e:
        print(f"Parsing failed: {str(e)}")
        return None
    
    return result

def predict_top_k_words(first_100_words, wordNum, statDict, datasetInfo="Educational story targeted at young children using simple words.  ", k=10):
    prompt = f"""
    You are a text analysis assistant specializing in corpus linguistics. 
    Your task is to estimate Top-{k} word frequency distribution based on partial corpus data.
    Here's the  Corpus Information:
    - Total word count: {wordNum}
    - Corpus type: {datasetInfo}
    - Sample words: {first_100_words}
    Return your answer as a JSON object with this structure:
    {{
        "top_elements": {{...}},  // Top-{k} word:frequency pairs, ordered in descending frequency
    }}
    Analysis Requirements:
    1. Generate estimated word frequency distribution
    2. Identify significant linguistic patterns
    3. Detect potential domain-specific terminology
    4. Assess sampling representativeness
    Only return valid JSON, **NO other text**. Example for k=3:
     Output Format:
    ```json
    {{
        "top_elements": {{"the": 9876, "and": 5432, "of": 4321}},
    }}
    ```
    """
    print("Prompt:",prompt)
    response = predict_single_Sil(prompt,model='Pro/deepseek-ai/DeepSeek-V3')
    response=response.replace('```json','')
    response=response.replace('```','')
    predicted_words = parse_model_output(response)
    
    return predicted_words

def predict_certain_words(  wordNum, specified_Word, datasetInfo="Educational story targeted at young children using simple words.  ",LLM_Name='Pro/Qwen/Qwen2.5-Coder-7B-Instruct',Hint=""):
    prompt = f"""
    You are a text analysis assistant specializing in corpus linguistics. 
    Your task is to estimate the frequency of Token **{specified_Word}** in the Corpus  based on the below corpus statistics and NL Description.
    Here's the Corpus Information:
    - Estimate goal: The Token {specified_Word}
    - Total Token count: {wordNum}
    - Corpus Description: {datasetInfo}
    - Hint Rule: {Hint}.
    Return your answer as a int object with this structure:
    Only return the frequency integer, **NO OTHER TEXT**.\\nothink"""
    response = predict_single_Sil(prompt,model=LLM_Name)

    return int(response)

def top_k_words(text, k):
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    actual_k = min(k, len(word_counts))
    return dict(word_counts.most_common(actual_k))

def bottom_k_words(text, k):
    words = re.findall(r'\b\w+\b', text.lower())
    word_counts = Counter(words)
    return dict(word_counts.most_common()[-k:][::-1])

def calculate_qerror_metrics(qerrors):
    """Calculate percentiles of Q-error distribution"""
    metrics = {
        'median': np.percentile(qerrors, 50),
        'p90': np.percentile(qerrors, 90),
        'p95': np.percentile(qerrors, 95),
        'p99': np.percentile(qerrors, 99),
        'max': np.max(qerrors)
    }

    return metrics

def compute_recall(predictFreq, actualFreq, k):
    """Compute recall@k - ratio of correctly predicted top-k words"""
    # Get top k words from actual frequency
    true_top_k = set(sorted(actualFreq.keys(), key=lambda x: -actualFreq[x])[:k])
    # Get top k words from predicted frequency
    predicted_top_k = set(sorted(predictFreq.keys(), key=lambda x: -predictFreq[x])[:k])
    # Compute intersection
    correct = true_top_k.intersection(predicted_top_k)
    return len(correct) / k

def compute_qerror(predictFreq, actualFreq):
    """Compute Q-error for top-n frequent words (quantifies prediction accuracy)"""
    qerrors = []
    for word in actualFreq.keys():
        if not (word in predictFreq.keys()):
            predicted =1
        else:
            predicted = predictFreq[word]
        actual = actualFreq[word]
        qerror = max((predicted+1)/(actual+1), (actual+1)/(predicted+1))
        qerrors.append(qerror)
    return calculate_qerror_metrics(np.array(qerrors))

def convert_keys_to_lower(input_dict):
    
    if not isinstance(input_dict, dict):
        raise TypeError("The input must be a dictionary")
    
    return {key.lower(): value for key, value in input_dict.items()}
    
def cureD(d1,d2):
    y1 = []
    y2 = []
    for ki in d1.keys():
        if (ki in d2.keys()):
            y1.append(d1[ki])
            y2.append(d2[ki])
    y1= np.array(y1) 
    y2= np.array(y2) 
    bias = sum(y1-y2)/len(y1)    
    return bias

def evaluate_predictionSingle(sl, args):

    realDict = sl[0]
    wordNum = realDict['<total>']
    realDict.pop('<total>', None)
    realDict.pop('<numwords>', None)

    sorted_keys = [key for key, value in sorted(realDict.items(), key=lambda item: item[1], reverse=True)]
    testKeys = sorted_keys[:args.TopKN_Eval]

    sampleDict =  sl[1]
    realSampleRatio = sampleDict['<sel>']
    
    qerrList = []
    predictFreq = {}
    actualFrep = {}
    print("---"*30)
    print("LLM_Name",args.LLM_Name)
    
    for word in testKeys:
        try:
            freq = realDict[word]
            actualFrep[word] = freq
            wordNumInput = wordNum
            if word in sampleDict.keys():
                sampledCount = sampleDict[word]
            else:
                sampledCount = 0

            if args.SampleRate == 0:
                Hint = "None"
            else:
                if sampledCount == 0:
                    Hint = f"The result is likely to between between 100 and {int(1/realSampleRatio)}"
                else:
                    Hint = f"The estimation from sampling is {int(sampledCount/realSampleRatio)}. Note that if the sample estimate is non-zero, it is likely that your estimation result will be close to with the sampling estimation. However, if the sampling estimate is 0, you should provide a guess between 0 and {int(1/realSampleRatio)} based on the natural language description of data, rather than simply repeating the estimated value of the sampling!"
            predictInt = predict_certain_words(wordNum=wordNumInput, specified_Word=word, datasetInfo=args.datasetDescribtion,LLM_Name=args.LLM_Name,Hint=Hint)
 
            qerr = max((predictInt+1)/(freq+1), (freq+1)/(predictInt+1))
            qerrList.append(qerr)

            predictFreq[word] = predictInt
        except Exception as e:
            print(f"Error processing word '{word}': {str(e)}. Skipping...")
            continue

    rectify_words = sampleDict
    rectify_words.pop('<total>', None)
    rectify_words.pop('<numwords>', None)
    
    rectify_Dict_raw =  heapq.nlargest(args.TopKN_Cure, rectify_words.items(), key=lambda x: x[1])
    rectify_Dict = {}
    for ki,freqx in rectify_Dict_raw:
        rectify_Dict[ki] = freqx / args.SampleRate
    
    bias = cureD(predictFreq, rectify_Dict)
    for ki in predictFreq.keys():
        predictFreq[ki] = predictFreq[ki] - bias 

    q_error = compute_qerror(predictFreq, actualFrep)
    print(f"Q-errors: {q_error }")
    print("---"*30)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Sketch Project")
    parser.add_argument('--datasetName', type=str, default="Children", help='nameOfDataset.')
    parser.add_argument('--datasetDescribtion', type=str, default="This is a Hugging face dataset **ajibawa-2023/Children-Stories-Collection** containing educational stories targeted at young children with 800K training Entries,, using simple words and phrases.", help='datasetDescribtion')
    parser.add_argument('--Num_In', type=int, default=800000)
    parser.add_argument('--SampleRate', type=float, default=0.00001)
    parser.add_argument('--TopKN_Eval', type=int, default=256)
    parser.add_argument('--TopKN_Cure', type=int, default=16)
    parser.add_argument('--LLM_Name', type=str, default="Qwen/Qwen2.5-32B-Instruct", help='nameOfLLM')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    setup_seed(args.seed)
    print_args(args)

    if args.datasetName =='Children':
        FullDict = load_dict_from_json('Children_Full')
        sampleDict = load_dict_from_json('Children_Sample1e-5')

    elif args.datasetName == 'Finance':
        FullDict = load_dict_from_json('Finance_Full')
        sampleDict = load_dict_from_json('Finance_Sample1e-5')

    elif args.datasetName == 'Amazon':
        FullDict = load_dict_from_json('Amazon_Full')
        sampleDict = load_dict_from_json('Amazon_Sample1e-5')

    elif args.datasetName == 'Math':
        FullDict = load_dict_from_json('Math_Full')
        sampleDict = load_dict_from_json('Math_Sample1e-5')

    infoList=[]
    infoList.append(FullDict)
    infoList.append(sampleDict)
    
    evaluate_predictionSingle(infoList, args)
