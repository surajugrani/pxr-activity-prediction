Experiments 
01: ChemProp Baseline - Just bare bones ChemProp with all default options to get a baseline performance
02: ChemProp using scaffold-balanced split for training
03: 2-model ensemble of ChemProp baseline and RF using Mordred Descriptors

|#|Description|MAE|RAE|R2|Spearman ρ|Kendall's τ|
|-|-|-|-|-|-|-|
|01|ChemProp default|0.4993|0.6273|0.5516|0.7854|0.5837|
|02|ChemProp scaffold balanced split|<br />0.5128 <br />|0.6444|0.5564|0.7798|0.5792|
|03|ChPr default + Mordred RD |0.5306|0.6666|0.4954|0.7759|0.5736|



