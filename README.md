Large Language Models (LLMs) have achieved enormous success in recently due to their ability to understand and solve various non-trivial tasks in natural language. However, they have been shown to memorize their training data \cite{carlini2019secret} which, among other concerns, increases risk of the model regurgitating creative or private content, potentially leading to legal issues for the model developer and/or vendors \cite{nytimeslawsuit}.  Often such issues are discovered post model training during testing or red teaming. Furthermore, stakeholders may sometimes request to remove their data after model training to protect copyright, or exercise their right to be forgotten \cite{GDPR}. In these instances, retraining models after discarding such data is one option but doing so after each such removal request is prohibitively expensive. \textit{Machine unlearning} is a relatively new research domain in machine learning which addresses this exact problem. 

While unlearning has been studied for sometime in classification problems, it is still a relatively underdeveloped area of study in LLM research since the latter operate in a potentially unbounded output label space. Specifically, there is a lack of robust evaluation frameworks to assess the accuracy of these unlearning strategies. In this challenge, we aim to bridge this gap by developing a comprehensive evaluation challenge for unlearning sensitive datasets in LLMs.


### Organizers
- Anil Ramakrishna, Amazon AGI
- Kai-Wei Chang, UCLA/Amazon AGI
- Rahul Gupta, Amazon AGI
- Volkan Cevher, EPFL/Amazon AGI
- Bhanu Vinzamuri, Amazon AGI
- He Xie, Amazon AGI
- Venkatesh Elango, Amazon AGI
- Woody Bu, Amazon AGI


### Contact
If you have any questions, please contact us at trustworthyspeechproc@gmail.com.
