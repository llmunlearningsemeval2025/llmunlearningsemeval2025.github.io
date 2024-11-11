### Important Updates
- July 17, 2024: Task announced
- September 2, 2024: Task artifacts are now live!
- November 10, 2024: Challenge awards announced along with benchmark results. 1B model released. 

### Overview

Large Language Models (LLMs) have achieved enormous success recently due to their ability to understand and solve various non-trivial tasks in natural language. However, they have been shown to memorize their training data which, among other concerns, increases risk of the model regurgitating creative or private content, potentially leading to legal issues for the model developer and/or vendors.  Often, such issues are discovered post model training during testing or red teaming. Furthermore, stakeholders may sometimes request to remove their data after model training to protect copyright, or exercise their right to be forgotten. In these instances, retraining models after discarding such data is one option but doing so after each such removal request is prohibitively expensive. Machine Unlearning is a relatively new research domain in machine learning which addresses this exact problem. 

While unlearning has been studied for sometime in classification problems, it is still a relatively underdeveloped area of study in LLM research since the latter operate in a potentially unbounded output label space. Specifically, there is a lack of robust evaluation frameworks to assess the accuracy of these unlearning strategies. In this challenge, we aim to bridge this gap by developing a comprehensive evaluation challenge for unlearning sensitive datasets in LLMs.

### [NEW] Announcing Awards for Top 3 submissions

The highest three performing submissions will be awarded cash prizes of $1000, $750 and $500 respectively. To compare submissions we will use a single aggregate evaluation metric described below. 

### Task Description

Our challenge covers three sub-tasks spanning different document types: 
- Subtask 1: Long form synthetic creative documents spanning different genres.
- Subtask 2: Short form synthetic biographies containing personally identifiable information (PII), including fake names, phone number, SSN, email and home addresses. 
- Subtask 3: Real documents sampled from the target model's training dataset. 

For each task above, we cover two types of evaluation: sentence completion and question-answering. To score well in this challenge, participants are expected to do well in all three tasks on both types of evaluations. 

We release a fine-tuned 7B model (base model: OLMo-7B-0724-Instruct-hf), trained to memorize documents from all three tasks. For each subtask, we also release specific Retain (i.e. model should retain these documents in memory) and Forget sets (i.e. model should forget these documents) along with the target model. Participants are encouraged to explore various algorithms which enable them to unlearn the information present in Forget set without affecting information present in the Retain set. Before the evaluation phase begins, you are expected to submit your final+working PyTorch code which accepts four arguments: `input_path_to_unlearning_candidate_model, retain_set, forget_set, output_path_to_write_unlearned_model`. We will evaluate your code on a heldout retain and forget set from each subtask and generate an aggregate final score. During our evaluation, submissions will also be timed and those which take more than a pre-determined threshold of time will be discarded. 

### Task Artifacts

Our fine-tuned model and dataset are now available to download! To get access, please complete this [registration form](https://tiny.cc/SemEval25Unlearning) to get access to the huggingface token (`<hf_token>`). Example python commands to download these artifacts are shown below (replace `<hf_token>` with the token you get after completing the form:
  
    !pip install --upgrade transformers huggingface_hub; mkdir semeval25-unlearning-model; mkdir semeval25-unlearning-data
    import pandas as pd
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer
    hf_token = <hf_token>   # Copy token here
    
    ## Fetch and load model:
    snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-finetuned-semeval25-unlearning', token=hf_token, local_dir='semeval25-unlearning-model')
    model = AutoModelForCausalLM.from_pretrained('semeval25-unlearning-model')
     
    ## Fetch and load dataset:
    snapshot_download(repo_id='llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public', token=hf_token, local_dir='semeval25-unlearning-data', repo_type="dataset")
    retain_train_df = pd.read_parquet('semeval25-unlearning-data/data/retain_train-00000-of-00001.parquet', engine='pyarrow') # Retain split: train set
    retain_validation_df = pd.read_parquet('semeval25-unlearning-data/data/retain_validation-00000-of-00001.parquet', engine='pyarrow') # Retain split: validation set
    forget_train_df = pd.read_parquet('semeval25-unlearning-data/data/forget_train-00000-of-00001.parquet', engine='pyarrow') # Forget split: train set
    forget_validation_df = pd.read_parquet('semeval25-unlearning-data/data/forget_validation-00000-of-00001.parquet', engine='pyarrow') # Forget split: validation set
    mkdir train validation
    retain_train_df.to_json('train/retain.jsonl'); forget_train_df.to_json('train/forget.jsonl')
    retain_validation_df.to_json('validation/retain.jsonl'); forget_validation_df.to_json('validation/forget.jsonl')
    
The dataset contains disjoint retain and forget splits in parquet files, and includes following fields: `id`, `input`, `output`, `task`. Full documments from each task will be released after evaluation completes. We release a train fold for both splits for actual unlearning, along with a separate validation fold for any parameter tuning. 

#### [NEW] Announcing a 1-Billion parameter model

Based on recent interest, we are releasing a new, smaller 1B LLM which is also fine-tuned to memorize the dataset in our unlearning benchmark similar to the 7B model. You can access this model similar to the 7B model and the `<hf_token>` using this id: `llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning`. 

    ## Fetch and load model:
    snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning', token=hf_token, local_dir='semeval25-unlearning-1B-model')
    model = AutoModelForCausalLM.from_pretrained('semeval25-unlearning-1B-model')
    
You're welcome to develop and tune your algorithms entirely on this 1B model or in addition to the 7B model. Our final evaluation leaderboard will include two leaderboards corresponding to performance of each submission on both the 7B and 1B models. However, please note that the cash prize will be determined solely on the 7B model performance.

### [NEW] Challenge Evaluation

To evaluate each submission, we compute *task specific* regurgitation rates (measured using `rouge-L` scores) on the sentence completion prompts and exact match rate for the question answers on both retain and forget sets; we invert forget set metrics to 1 - their value. In addition, we compute i) Membership Inference Attack (MIA) rates using loss based attack on a sample of member+nonmember datasets, and ii) compute model performance on the MMLU benchmark. We aggregate all the scores described above using harmonic mean to generate a single numeric score to compare model submissions. We're releasing our evaluation script with the data repository. You can download the evaluation script along with the MIA dataset from the repository `'llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public'` using commands listed above. 

### Unlearning Benchmark

Upcoming on 11/11/2024

### Challenge Platform

To be announced.

### Sample Data

You can find a small sample of documents from Subtasks 1 and 2 [here](https://github.com/llmunlearningsemeval2025/sample-data).

### Organizers

- Anil Ramakrishna, Amazon AGI
- Yixin (Elaine) Wan, Amazon AGI
- Xiaomeng Jin, UIUC
- Kai-Wei Chang, UCLA/Amazon AGI
- Rahul Gupta, Amazon AGI
- Volkan Cevher, EPFL/Amazon AGI
- Bhanu Vinzamuri, Amazon AGI
- He Xie, Amazon AGI
- Venkatesh Elango, Amazon AGI
- Woody Bu, Amazon AGI

### Important Dates

- Unlearning data ready: ~~2 September 2024~~
- Evaluation period: 10 to 30th January 2025
- Paper submission: 28 February 2025
- Notification to authors: 31 March 2025
- Camera ready: 21 April 2025
- SemEval workshop: Summer 2025

All deadlines are 23:59 UTC-12 ("anywhere on Earth").

### Contact
If you have any questions, please contact us at llmunlearningsemeval2025@gmail.com. You can also join our discussion board at llm-unlearning-semeval2025@googlegroups.com. 
