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
