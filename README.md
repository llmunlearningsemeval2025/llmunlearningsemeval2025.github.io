Large Language Models (LLMs) have achieved enormous success recently due to their ability to understand and solve various non-trivial tasks in natural language. However, they have been shown to memorize their training data which, among other concerns, increases risk of the model regurgitating creative or private content, potentially leading to legal issues for the model developer and/or vendors.  Often, such issues are discovered post model training during testing or red teaming. Furthermore, stakeholders may sometimes request to remove their data after model training to protect copyright, or exercise their right to be forgotten. In these instances, retraining models after discarding such data is one option but doing so after each such removal request is prohibitively expensive. Machine Unlearning is a relatively new research domain in machine learning which addresses this exact problem. 

While unlearning has been studied for sometime in classification problems, it is still a relatively underdeveloped area of study in LLM research since the latter operate in a potentially unbounded output label space. Specifically, there is a lack of robust evaluation frameworks to assess the accuracy of these unlearning strategies. In this challenge, we aim to bridge this gap by developing a comprehensive evaluation challenge for unlearning sensitive datasets in LLMs.

Our challenge task covers three training data types listed below: 
- Long form synthetic creative documents
- Short form synthetic biographies containing personally identifiable information (PII)
- Real text snippets

For data type listed above, we provide sentence completion and question answer tasks, leading to six unique tasks covered in this challenge. 

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
