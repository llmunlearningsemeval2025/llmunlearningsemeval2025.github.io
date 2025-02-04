### Important Updates
- July 17, 2024: Task announced
- September 2, 2024: Task artifacts are now live!
- November 10, 2024: Challenge awards announced along with release of a 1B model.
- December 3, 2024: 7B Benchmark results added to website along with an update to evaluation script.
- December 26, 2024: Code submission form for challenge evaluation is live! More details [below](https://llmunlearningsemeval2025.github.io/#challenge-evaluation-phase).
- January 7, 2025: Important updates about evaluation process. 
- February 3, 2025: 7B leaderboard announced. Congratulations to the award winners!

### Overview

Large Language Models (LLMs) have achieved enormous success recently due to their ability to understand and solve various non-trivial tasks in natural language. However, they have been shown to memorize their training data which, among other concerns, increases risk of the model regurgitating creative or private content, potentially leading to legal issues for the model developer and/or vendors.  Often, such issues are discovered post model training during testing or red teaming. Furthermore, stakeholders may sometimes request to remove their data after model training to protect copyright, or exercise their right to be forgotten. In these instances, retraining models after discarding such data is one option but doing so after each such removal request is prohibitively expensive. Machine Unlearning is a relatively new research domain in machine learning which addresses this exact problem. 

While unlearning has been studied for sometime in classification problems, it is still a relatively underdeveloped area of study in LLM research since the latter operate in a potentially unbounded output label space. Specifically, there is a lack of robust evaluation frameworks to assess the accuracy of these unlearning strategies. In this challenge, we aim to bridge this gap by developing a comprehensive evaluation challenge for unlearning sensitive datasets in LLMs.

### Announcing Awards for Top 3 submissions

The highest three performing submissions will be awarded cash prizes of $1000, $750 and $500 respectively. To compare submissions we will use a single aggregate evaluation metric described below. 

### Task Description

Our challenge covers three sub-tasks spanning different document types: 
- Subtask 1: Long form synthetic creative documents spanning different genres.
- Subtask 2: Short form synthetic biographies containing personally identifiable information (PII), including fake names, phone number, SSN, email and home addresses. 
- Subtask 3: Real documents sampled from the target model's training dataset. 

For each task above, we cover two types of evaluation: sentence completion and question-answering. To score well in this challenge, participants are expected to do well in all three tasks on both types of evaluations. 

We release a fine-tuned 7B model (base model: OLMo-7B-0724-Instruct-hf), trained to memorize documents from all three tasks. For each subtask, we also release specific Retain (i.e. model should retain these documents in memory) and Forget sets (i.e. model should forget these documents) along with the target model. Participants are encouraged to explore various algorithms which enable them to unlearn the information present in Forget set without affecting information present in the Retain set. Before the evaluation phase begins, you are expected to submit your final+working PyTorch code which accepts four arguments: `input_path_to_unlearning_candidate_model, output_path_to_write_unlearned_model, path_to_forget_set, path_to_retain_set`. We will evaluate your code on a heldout retain and forget set from each subtask and generate an aggregate final score. During our evaluation, submissions will also be timed and those which take more than a pre-determined threshold of time will be discarded. 

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
    retain_train_df.to_json('train/retain.jsonl', orient='records', lines=True); forget_train_df.to_json('train/forget.jsonl', orient='records', lines=True)
    retain_validation_df.to_json('validation/retain.jsonl', orient='records', lines=True); forget_validation_df.to_json('validation/forget.jsonl', orient='records', lines=True)
    
The dataset contains disjoint retain and forget splits in parquet files, and includes following fields: `id`, `input`, `output`, `task`. Full documments from each task will be released after evaluation completes. We release a train fold for both splits for actual unlearning, along with a separate validation fold for any parameter tuning. 

#### Tokenizers
We use following (default) tokenizers for both models:
- OLMo-7B-0724-Instruct-hf: `allenai/OLMo-7B-0724-Instruct-hf`
- OLMo-1B-0724-hf: `allenai/OLMo-1B-0724-hf`

#### Announcing a 1-Billion parameter model

Based on recent interest, we are releasing a new, smaller 1B LLM which is also fine-tuned to memorize the dataset in our unlearning benchmark similar to the 7B model. You can access this model similar to the 7B model and the `<hf_token>` using this id: `llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning`. 

    ## Fetch and load model:
    snapshot_download(repo_id='llmunlearningsemeval2025organization/olmo-1B-model-semeval25-unlearning', token=hf_token, local_dir='semeval25-unlearning-1B-model')
    model = AutoModelForCausalLM.from_pretrained('semeval25-unlearning-1B-model')
    
You're welcome to develop and tune your algorithms entirely on this 1B model or in addition to the 7B model. Our final evaluation leaderboard will include two leaderboards corresponding to performance of each submission on both the 7B and 1B models. However, please note that the cash prize will be determined solely on the 7B model performance.

### Challenge Evaluation Phase
Form to submit your evaluation code is live! You can access the form [here](https://docs.google.com/forms/d/e/1FAIpQLSfyv10P5WvNwa-uRvx-VdOW1yscfU05psND96vcHqxegR0btQ/viewform?usp=preview).
- **Timeline**: The evaluation period is from January 10th to 30th. Please target submitting your code before January 10th to give us enough time to evaluate all submissions (we will leave the submission form open until January 15th for a buffer but we strongly encourage you to submit early). 
- **Submission File**: Please prepare your submission in the form of a single python (.py) file. You can submit your code file via the form shown above.
  - Your submission file should include a **single function named unlearn** which accepts following arguments as input, conducts unlearning and stores output model checkpoints:
    - Path to the fine tuned model to read from (includes the tokenizer).
    - Path to the output directory to write the unlearned checkpoints to. 
    - Path for the private forget dataset directory (contains both jsonl/parquet files as `forget.parquet` and `forget.jsonl`).
    - Path for the private retain dataset directory (contains `retain.parquet` and `retain.jsonl`).
- **Evaluation Infrastructure**: Your code will be executed on an AWS EC2 p4d.24xlarge node with limited execution permissions. To be fair for all, every submission will be timed out after one hour so please ensure your code stores relevant model checkpoints in the target path frequently. 
  - It is important you **store a single best checkpoint** in the target path which we will use for final evaluations. 
  - We will use Python 3.12.8, with latest versions of packages listed [here](https://github.com/llmunlearningsemeval2025/llmunlearningsemeval2025.github.io/blob/main/requirements.txt) pre-installed. If you need us to include any additional packages with your evaluation, you can list this in the code submission form. The training environment will be configured with DeepSpeed Zero-3 if you wish to leverage model distributed training with default parameters; you may modify these parameters via HuggingFace accelerate's deep-speed plugin.
  - Due to limited compute, at this time were only able to accept one submission per team. Please refrain from making multiple submissions: we will select the most recent submission for evaluation. 
  - The time limit of 1 hour was determined by running gradient difference using the publicly available [TOFU codebase](https://github.com/locuslab/tofu) for 10 epochs (with batch size = 32, learning rate = 1e-5), with an added 15 minute buffer time. We **strongly** encourage you to test your code locally to ensure you are able to obtain desired performance within the specified time frame (you can also use the TOFU code base to run gradient difference for 10 epochs for reference). 
- **MMLU**: We will evaluate the model by obtaining completion probabilities for all four options and select the choice with highest probability. We will leverage the Open Instruct MMLU evaluation code base for this task. 
- **Evaluation Metric**: As a reminder, we will use the evaluation strategy (and script) described below to compute the final metric for both 1B and 7B models.

#### Evaluation Metric

To evaluate each submission, we compute following metrics:
1. *task specific* regurgitation rates (measured using `rouge-L` scores) on the sentence completion prompts and exact match rate for the question answers on both retain and forget sets; we invert forget set metrics to 1 - their value. We aggregate all 12 distinct scores described above to generate a single numeric score via harmonic mean.
2. A Membership Inference Attack (MIA) score using loss based attack on a sample of member+nonmember datasets, given by: `1 - abs(mia_loss_auc_score - 0.5)*2`.
3. Model performance on the MMLU benchmark, measured as test accuracy on 57 STEM subjects.
   - For the awards leaderboard, we only consider solutions for which the MMLU accuracy is **above 0.371** (75% of the pre-unlearning checkpoint) - this ensures we do not degrade model utility due to unlearning. 

Finally, we aggregate all three scores described above to generate a single numeric score to compare model submissions, using arithmetic mean. Please use the official evaluation script to estimate your model performance. You can download this script along with the MIA dataset from the repository `'llmunlearningsemeval2025organization/semeval25-unlearning-dataset-public'` using commands listed above. 

### Leaderboard

We received nearly 100 submissions from 26 teams for this challenge. We thank all the teams for their interest and a strong participation in our challenge. Final leaderboard for the 7B model is shown below (the final score column represents arithmetic mean of next three columns):

#### 7 Billion parameter model leaderboard:

| Team | Final Score | Task Aggregate | MIA Score | MMLU Avg. | 
| :--------- | :---------: | :-------------: | :---: | :----: |
| 1. **AILS-NTUA** | 0.706 | 0.827 | 0.847 | 0.443 |
| 2. **ZJUKLAB** | 0.487 | 0.944 | 0.048 | 0.471 |
| 3. **ch******3@stu.ynu.edu.cn** | 0.470 | 0.834 | 0.139 | 0.436 |
| 4. Mr. Snuffleupagus | 0.376 | 0.387 | 0.256 | 0.485 |
| 5. su******4@gmail.com | 0.326 | 0.496 | 0.0 | 0.481 |
| 6. hk**@georgetown.edu | 0.3082 | 0.433 | 0.0 | 0.492 |
| 7. GIL-IIMAS UNAM  | 0.3080 | 0.478 | 0.0 | 0.446 |
| 8. Atyaephyra | 0.273 | 0.348 | 0.014 | 0.456 |
| 9. Lacuna Inc. | 0.251 | 0.283 | 0.0 | 0.469 |
| 10. APT | 0.169 | 0.1 | 0.021 | 0.385 |
| 11. h.s***********0@gmail.com | 0.165 | 0.0 | 0.0 | 0.495 |
| 11. JU-CSE-NLP'25 | 0.165 | 0.0 | 0.0 | 0.495 |
| 13. Tsotsa  | 0.1649 | 0.0 | 0.0 | 0.495 |
| 14. ma********8@gmail.com | 0.154 | 0.0 | 0.0 | 0.463 |

Honarary mention for submissions not included in awards consideration list above since the MMLU scores dropped below the predefined threshold of 0.371:

| Team | Final Score | Task Aggregate | MIA Score | MMLU Avg. | 
| :--------- | :---------: | :-------------: | :---: | :----: |
| SHA256 | 0.711 | 0.964 | 0.894 | 0.275 |
| NeuroReset | 0.420 | 0.152 | 0.876 | 0.232 |
| Cyber for AI | 0.409 | 0.0 | 0.999 | 0.229 |
| MALTO | 0.402 | 0.0 | 0.965 | 0.242 |
| Innovative_Team | 0.365 | 0.0 | 0.849 | 0.247 |
| ay***********0@gmail.com | 0.356 | 0.0 | 0.84 | 0.229 |

#### 1 Billion parameter model leaderboard:
Upcoming.

### Benchmark of unlearning algorithms on dataset

| Algorithm | Aggregate | Task Aggregate | MIA Score | MMLU Avg. | 
| :--------- | :---------: | :-------------: | :---: | :----: |
| ~~Gradient Ascent~~ | ~~0.394~~ | ~~0~~ | ~~0.912~~ | ~~0.269~~ |
| Gradient Difference | 0.243 | 0 | 0.382 | 0.348 |
| ~~KL Minimization~~ | ~~0.395~~ | ~~0~~ | ~~0.916~~ | ~~0.269~~ |
| Negative Preference Optimization | 0.188 | 0.021 | 0.080 | 0.463 |

Gradient Ascent and KL Minimization were discarded since they severely degrade model utility (MMLU drops below predetermined threshold of 0.371).

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
- Zhiqi (Woody) Bu, Amazon AGI
- Mingyi Hong, Univ. of Minnesota/Amazon AGI

### Important Dates

- Unlearning data ready: ~~2 September 2024~~
- Evaluation period: ~~10 to 30th January 2025~~
- Paper submission: 28 February 2025
- Notification to authors: 31 March 2025
- Camera ready: 21 April 2025
- SemEval workshop: 31 July - 1 August 2025 (co-located with [ACL 2025](https://2025.aclweb.org/))

All deadlines are 23:59 UTC-12 ("anywhere on Earth").

### Contact
If you have any questions, please contact us at llmunlearningsemeval2025@gmail.com. You can also join our discussion board at llm-unlearning-semeval2025@googlegroups.com. 
