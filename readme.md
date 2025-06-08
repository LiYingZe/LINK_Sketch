# 🚀 **The Zero-Shot Semantic Sketch**

Approximate frequency counting plays a vital role in analyzing large-scale corpora 📚, especially in the context of training and curating datasets for Large Language Models (LLMs) 🤖. Traditional approaches—such as sketch-based and sampling-based methods—often struggle in enterprise-scale settings due to high computational overhead 💻 or poor accuracy when data access is limited 🔒.

Recent advances in LLMs have opened up new possibilities for frequency estimation using only minimal data access and natural language dataset descriptions 💡. However, LLMs are prone to hallucinations 🌀 and inherent biases 🎭, which can significantly undermine the reliability of their estimates—particularly when applied to domain-specific or niche corpora 🎯.

------

### 🔗 Meet **LINK**: *Language-Informed Neural Knowledge* 🧠✨

![LINK](./LINK.png)

LINK is a novel estimator that combines **sparse data-level statistics** 📊 with **semantic-level dataset descriptions** 📝 for comprehensive frequency estimation. Data-level augmented signals serve as unbiased anchors ⚓ for both enhancement signals and hallucinations, effectively reducing LLM hallucinations and biases 🛡️. Semantic-level dataset summarization provides condensed features beyond sparse data 🗜️, offering additional global frequency information that enables more accurate frequency estimation 🎯.

### 🏆 Why LINK?

- 🎯 **Unbiased Anchoring**: Sparse data-level statistics eliminate LLM hallucinations
- 🧠 **Semantic Intelligence**: Natural language descriptions provide global context
- 📊 **Enterprise-Scale**: Handles massive corpora with minimal computational overhead
- 🔒 **Limited Access Friendly**: Works efficiently with restricted data access
- 🎭 **Bias Reduction**: Effectively mitigates inherent LLM biases
- ⚡ **Zero-Shot Capability**: No training required on target datasets

------

## 🗂️ Code Structure

Our project is organized as follows:

```
.
├── data                           📁 Dataset collection
│   ├── Amazon                     🛒 E-commerce dataset
│   │   ├── download.py            📥 Download script
│   │   └── parse.py               🔍 Parser utility
│   ├── Children                   👶 Children's literature dataset
│   │   ├── download.py            📥 Download script
│   │   └── parse.py               🔍 Parser utility
│   ├── Finance                    💰 Financial corpus dataset
│   │   ├── download.py            📥 Download script
│   │   └── parse.py               🔍 Parser utility
│   └── Math                       🔢 Mathematical text dataset
│       ├── download.py            📥 Download script
│       └── parse.py               🔍 Parser utility
├── cache                          💾 Processed frequency dictionaries
├── main.py                        🚀 Main evaluation program
├── interactive_chat.py            💬 Interactive query interface
├── generate_cache.py              🛠️ Cache generation utility
├── dataset_config.txt             📋 Dataset configuration and statistics
└── requirements.txt               📋 Dependencies list
```

------

## 📦 Requirements

- 🔄 python==3.11.0
- 📚 datasets==2.19.0
- 🤗 huggingface-hub==0.27.1
- 🔢 numpy==1.26.4
- 🤖 openai==1.74.0

------

## 📥 Setup & Dataset

We provide **4 HuggingFace corpus datasets** (Amazon, Children, Finance, Math) 🎯. To use them, you first need to download the datasets from HuggingFace 🤗.

### 📚 Dataset Download

Taking the **Children** dataset as an example, use the following commands:

```bash
huggingface-cli login    # 🔐 Login to your HuggingFace account for access
python ./data/Children/download.py    # 📥 Download dataset from HuggingFace
```

✅ **Repeat this process** for other datasets (Amazon, Finance, Math) by navigating to their respective directories! 🔄

------

## 💾 Generate Dataset Cache

Parse the downloaded datasets and convert files into frequency dictionaries 📊 for convenient subsequent operations. Taking the **Children** dataset as an example, use the following commands:

```bash
# 📈 Get complete frequency information
python ./generate_cache.py --datasetName "Children" --Num_In 800000 --SampleRate 1 --SaveName "Children_Full"

# 🎲 Get sampled frequency information  
python ./generate_cache.py --datasetName "Children" --Num_In 800000 --SampleRate 1e-5 --SaveName "Children_Sample1e-5"
```

🎉 **Now your dataset is loaded and parsed!** Ready for the next steps! 🚀

------

## 🧪 Test LINK

Our main program can be used to test LINK's estimation performance on different datasets 📊, using **Q-error** as the evaluation metric 📏. You need to specify dataset parameters (`--datasetName`, `--Num_In`, `--datasetDescription`) in the arguments or use default parameters 🛠️. All dataset statistical information is available in the `dataset_config.txt` file 📋.

Run the following command with default parameters:

```bash
python ./main.py
```

Or customize with specific dataset parameters:

```bash
python ./main.py --datasetName "Children" --Num_In 800000 --datasetDescription "..."
```

🔍 This evaluates LINK's frequency estimation accuracy across configured datasets and provides comprehensive performance metrics! All dataset configurations and statistical information can be found in `dataset_config.txt` 📈✨

------

## 💬 Interactive Word Frequency Prediction 🤖🔍

We provide an **interactive program** 🕹️ that supports specific word frequency queries 🔎. Simply add the `--word` parameter in the command line to predict word frequency through LINK 🎯. You also need to specify which dataset to use in the parameters 📋.

### 🧙 Try it Yourself!

Taking the most common word **"the"** as an example with the Children dataset:

```bash
python ./interactive_chat.py --word "the" --datasetName "Children"
```

Or test with different datasets:

```bash
python ./interactive_chat.py --word "investment" --datasetName "Finance" --Num_In 518815 --datasetDescription "..."
python ./interactive_chat.py --word "algorithm" --datasetName "Math" --Num_In 6497564 --datasetDescription "..."
```

------

### 🎮 What it does:

- 🗣️ **Converse with LINK**: Effortlessly query word frequencies in natural language
- ⚡ **Instantaneous Results**: Get frequency predictions in real-time
- 🎯 **Semantic Understanding**: Leverages dataset descriptions for accurate estimation
- 💡 **Zero-Shot Magic**: No training required on target datasets
- 📊 **Enterprise-Scale**: Handles massive corpora efficiently
- 🛡️ **Bias-Free**: Reduces LLM hallucinations with data-level anchoring

------

### 🎯 Sample Queries to Try

Here are some interesting words you can test:

- `"artificial"` - Technology domain 🤖
- `"investment"` - Finance domain 💰  
- `"learning"` - Education domain 📚
- `"algorithm"` - Computer science domain 💻
- `"children"` - Literature domain 👶

> 🧠 *Behind the scenes*: LINK combines sparse data statistics with semantic dataset understanding to provide accurate frequency estimates—**way smarter** than traditional sketch-based methods! 🔥

------

Whether you're analyzing corpus statistics, curating LLM datasets, or exploring word distributions—LINK makes frequency estimation **intelligent and reliable**! 🤓🎉

Ready to **link** your data with semantic intelligence? 🔗✨ Let the zero-shot frequency magic begin! ⚡🧠📊

------

**LINK** revolutionizes frequency estimation by bridging the gap between sparse data access and semantic understanding 🌉. Say goodbye to hallucinations and hello to **reliable, enterprise-scale frequency counting**! 🎯🚀

Ready to make your frequency estimates **smarter**? 😄 Let neural semantic sketching begin! ⚡🤖📈