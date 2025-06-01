<div align="center">
  
## Multi Stage Retrieval for Web Search during Crisis

[**Claudiu Tcaciuc**](https://github.com/ClaudiuTcaciuc)<sup>1</sup> · [**Daniele Rege Cambrin**](https://darthreca.github.io/)<sup>1</sup> · [**Paolo Garza**](https://dbdmg.polito.it/dbdmg_web/people/paolo-garza/)<sup>1</sup>

<sup>1</sup>Politecnico di Torino, Italy

<a href="https://arxiv.org/abs/2408.04523"><img src='https://img.shields.io/badge/MDPI-MultiStage%20Paper-blue?logo=openaccess' alt='Paper PDF'></a>
</div>

This paper introduces a novel multi-stage text retrieval framework to enhance information retrieval during crises. Our framework employs a novel **three-stage extractive pipeline** where (1) a **topic modeling** component filters candidates based on thematic relevance, (2) an initial **high-recall lexical retriever** identifies a broad candidate set, and (3) a **dense retriever** reranks the remaining documents. **Existing approaches strongly rely on the power of large language models. However, the use of large language models limits the scalability of the retrieval procedure and may introduce hallucinations.** Our sequential approach accelerates the search process by 5% compared to the use of a single-stage based on a dense retrieval approach.

## Getting Started

Install the dependencies of the *requirements.txt* file.

You can find the additional queries and the mapping with the events in the *queries* folder.

## License

This project is licensed under the **Apache 2.0 license**. See [LICENSE](LICENSE) for more information.

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{Tcaciuc2025,
  title = {Multi Stage Retrieval for Web Search During Crisis},
  volume = {17},
  ISSN = {1999-5903},
  url = {http://dx.doi.org/10.3390/fi17060239},
  DOI = {10.3390/fi17060239},
  number = {6},
  journal = {Future Internet},
  publisher = {MDPI AG},
  author = {Tcaciuc,  Claudiu Constantin and Rege Cambrin,  Daniele and Garza,  Paolo},
  year = {2025},
  month = may,
  pages = {239}
}
```
