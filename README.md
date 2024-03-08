# NMTF-with-auxiliary-information

2023年度日本分類学会シンポジウム発表内容　実装  
発表内容の詳細は[こちら](https://acrobat.adobe.com/id/urn:aaid:sc:AP:a72c501d-992d-4f01-b7da-c1a5b9eb1f31)  
Methods presented at the academic conference.  
See [here](https://acrobat.adobe.com/id/urn:aaid:sc:AP:a72c501d-992d-4f01-b7da-c1a5b9eb1f31) for details of the method.

### Prerequisites
***
- Python
- Numpy
- Pandas
- tqdm
- PyTorch
- TensorLy

### Introduction
***
Conventional Non-negative Multiple Tensor Factorization (NMTF) methods are unable to incorporate "additional information" that represents the underlying structure of the data, such as feature adjacencies, even if such information exists. This method addresses this issue by adding penalties that capture the continuous variation of data in a particular mode in order to incorporate additional information about spatio-temporal adjacencies. This method is expected to be applied to data analysis in various fields, as it allows for the clear extraction of hidden patterns in spatio-temporal data.

### References
***
[1] Ahn, D., Jang, J. G., & Kang, U. (2022). Time-aware tensor decomposition for sparse tensors. Machine Learning, 111(4), 1409-1430.  
[2] Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.  
[3] Rudin, L. I., Osher, S., & Fatemi, E. (1992). Nonlinear total variation based noise removal algorithms. Physica D: nonlinear phenomena, 60(1-4), 259-268.  
[4] Shashua, A., & Hazan, T. (2005, August). Non-negative tensor factorization with applications to statistics and computer vision. In Proceedings of the 22nd international conference on Machine learning (pp. 792-799).  
[5] Shin, K., Sael, L., & Kang, U. (2016). Fully scalable methods for distributed tensor factorization. IEEE Transactions on Knowledge and Data Engineering, 29(1), 100-113.  
[6] Takeuchi, K., Tomioka, R., Ishiguro, K., Kimura, A., & Sawada, H. (2013, December). Non-negative multiple tensor factorization. In 2013 IEEE 13th International Conference on Data Mining (pp. 1199-1204). IEEE.  
