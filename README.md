Large Language Models (LLMs) have achieved enormous success recently due to their ability to understand and solve various non-trivial tasks in natural language. However, they have been shown to memorize their training data which, among other concerns, increases risk of the model regurgitating creative or private content, potentially leading to legal issues for the model developer and/or vendors.  Often, such issues are discovered post model training during testing or red teaming. Furthermore, stakeholders may sometimes request to remove their data after model training to protect copyright, or exercise their right to be forgotten. In these instances, retraining models after discarding such data is one option but doing so after each such removal request is prohibitively expensive. Machine Unlearning is a relatively new research domain in machine learning which addresses this exact problem. 

While unlearning has been studied for sometime in classification problems, it is still a relatively underdeveloped area of study in LLM research since the latter operate in a potentially unbounded output label space. Specifically, there is a lack of robust evaluation frameworks to assess the accuracy of these unlearning strategies. In this challenge, we aim to bridge this gap by developing a comprehensive evaluation challenge for unlearning sensitive datasets in LLMs.

Our challenge covers three tasks spanning different document types listed below: 
- Task 1: Long form synthetic creative documents spanning various genres
- Task 2: Short form synthetic biographies containing personally identifiable information (PII)
- Task 3: Documents sampled from pretraining dataset

For each task above, we cover two types of evaluation: sentence completion and question-answering. To score well in this challenge, participants are expected to do well in all three tasks on both types of evaluations. 

### Sample Data

You can find a small sample of documents from Task 1 and Task 2 [here](https://github.com/llmunlearningsemeval2025/sample-data). 

### Challenge Evaluation Metrics

To be announced.

### Challenge Platform

To be announced.

###

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


### Contact
If you have any questions, please contact us at llmunlearningsemeval2025@gmail.com. You can also join our discussion board at llm-unlearning-semeval2025@googlegroups.com. 
