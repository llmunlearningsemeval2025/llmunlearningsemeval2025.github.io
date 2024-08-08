Large Language Models (LLMs) have achieved enormous success recently due to their ability to understand and solve various non-trivial tasks in natural language. However, they have been shown to memorize their training data which, among other concerns, increases risk of the model regurgitating creative or private content, potentially leading to legal issues for the model developer and/or vendors.  Often, such issues are discovered post model training during testing or red teaming. Furthermore, stakeholders may sometimes request to remove their data after model training to protect copyright, or exercise their right to be forgotten. In these instances, retraining models after discarding such data is one option but doing so after each such removal request is prohibitively expensive. Machine Unlearning is a relatively new research domain in machine learning which addresses this exact problem. 

While unlearning has been studied for sometime in classification problems, it is still a relatively underdeveloped area of study in LLM research since the latter operate in a potentially unbounded output label space. Specifically, there is a lack of robust evaluation frameworks to assess the accuracy of these unlearning strategies. In this challenge, we aim to bridge this gap by developing a comprehensive evaluation challenge for unlearning sensitive datasets in LLMs.

### Task Description

Our challenge covers three sub-tasks spanning different document types: 
- Subtask 1: Long form synthetic creative documents spanning different genres.
- Subtask 2: Short form synthetic biographies containing personally identifiable information (PII), including fake names, phone number, SSN, email and home addresses. 
- Subtask 3: Real documents sampled from the target model's training dataset. 

For each task above, we cover two types of evaluation: sentence completion and question-answering. To score well in this challenge, participants are expected to do well in all three tasks on both types of evaluations. 

We release a fine-tuned LLM (~3B to 7B parameters in size; exact model size to be released with dataset on September 2nd), which has memorized the documents from all three subtasks. For each subtask, we also release specific Retain (i.e. model should retain these documents in memory) and Forget sets (i.e. model should forget these documents) along with the target model. Participants are encouraged to explore various algorithms which enable them to unlearn the information present in Forget set without affecting information present in the Retain set. Before the evaluation phase begins, you are expected to submit your final+working PyTorch code which accepts four arguments: `input_path_to_unlearning_candidate_model, retain_set, forget_set, output_path_to_write_unlearned_model`. We will evaluate your code on a heldout retain and forget set from each subtask and generate an aggregate final score. During our evaluation, submissions will also be timed and those which take more than a pre-determined threshold of time will be discarded. 

### Sample Data

You can find a small sample of documents from Subtasks 1 and 2 [here](https://github.com/llmunlearningsemeval2025/sample-data). Sentence completion and QA samples for these two tasks along with samples from Subtask 3 will be made available on September 2nd, 2024. 

### Challenge Evaluation Metrics

To be announced.

### Challenge Platform

To be announced.

### Organizers

- Anil Ramakrishna, Amazon AGI
- Kai-Wei Chang, UCLA/Amazon AGI
- Rahul Gupta, Amazon AGI
- Volkan Cevher, EPFL/Amazon AGI
- Bhanu Vinzamuri, Amazon AGI
- He Xie, Amazon AGI
- Venkatesh Elango, Amazon AGI
- Woody Bu, Amazon AGI
- Elaine Wan, Amazon AGI
- Xiaomeng Jin, UIUC

### Important Dates

- Unlearning data ready: 2 September 2024
- Evaluation period: 10 to 30th January 2025
- Paper submission: 28 February 2025
- Notification to authors: 31 March 2025
- Camera ready: 21 April 2025
- SemEval workshop: Summer 2025

All deadlines are 23:59 UTC-12 ("anywhere on Earth").

### Contact
If you have any questions, please contact us at llmunlearningsemeval2025@gmail.com. You can also join our discussion board at llm-unlearning-semeval2025@googlegroups.com. 
