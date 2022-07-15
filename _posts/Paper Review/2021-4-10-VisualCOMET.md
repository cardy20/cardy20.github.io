VisualCOMET: Reasoning about the Dynamic Context of a Still Image

Jae Sung Park1\,2Chandra Bhagavatula2RoozbechMottaghi1\,2Ali  Farhadi1YejinChoi1\,2

1Paul G\. Allen School of Computer Science & Engineering\, WA\, USA

2Allen Institute for Artificial Intelligence\, WA\, USA

ECCV ‘20

SanheePark

carpediem20@korea\.ac\.kr

Data Intelligence Lab\, Korea University

2021\.03\.16

<img src="img/VisualCOMET0.png" width=500px />

<img src="img/VisualCOMET1.png" width=481px />

<img src="img/VisualCOMET2.png" width=500px />

<img src="img/VisualCOMET3.png" width=391px />

<img src="img/VisualCOMET4.png" width=500px />

<span style="color:#002060">Screen shot from movie</span>

<img src="img/VisualCOMET5.png" width=500px />

<span style="color:#002060">Object detection</span>

<span style="color:#002060">Scene graph</span>  <span style="color:#002060">\(</span>  <span style="color:#002060">johnson et\. al\.2015  krishna et\. al\.\,2016</span>  <span style="color:#002060">\)</span>

<img src="img/VisualCOMET6.png" width=500px />

<img src="img/VisualCOMET7.png" width=500px />

<img src="img/VisualCOMET8.png" width=128px />

<span style="color:#002060">A man is holding onto a bronze statue in water\.</span>

<img src="img/VisualCOMET9.png" width=500px />

<img src="img/VisualCOMET10.png" width=500px />

<img src="img/VisualCOMET11.png" width=500px />

<img src="img/VisualCOMET12.jpg" width=211px />

<img src="img/VisualCOMET13.png" width=128px />

<span style="color:#002060">A man is holding onto a bronze statue in water\.</span>

<img src="img/VisualCOMET14.png" width=500px />

<span style="color:#002060">Given a person in the image\,</span>

<span style="color:#002060">people can reason about the rich dynamic story underlying the visual scene</span>

<span style="color:#002060">Rich background knowledge about how visual world works\, how the social world works</span>

\[2\]Antol\, S\.\, Agrawal\, A\.\, Lu\, J\.\, Mitchell\, M\.\,Batra\, D\.\,Zitnick\, C\. L\.\, & Parikh\, D\. \(2015\)\.Vqa: Visual question answering\. In Proceedings of the IEEE international conference on computer vision \(pp\. 2425\-2433\)\.

<img src="img/VisualCOMET15.jpg" width=500px />

<span style="color:#002060"> __Visual Question Answering__ </span>  <span style="color:#002060"> __\[2\]__ </span>

<img src="img/VisualCOMET16.jpg" width=500px />

<span style="color:#002060"> __Image Captioning__ </span>  <span style="color:#002060"> __\[1\]__ </span>

<span style="color:#002060"> __ Fall short of understanding the dynamic situation\.__ </span>

\[1\] Lin\, T\. Y\.\,Maire\, M\.\,Belongie\, S\.\, Hays\, J\.\,Perona\, P\.\,Ramanan\, D\.\, \.\.\. &Zitnick\, C\. L\. \(2014\, September\)\. Microsoftacoco: Common objects in context\. In European conference on computer vision \(pp\. 740\-755\)\. Springer\, Cham\.

<span style="color:#002060"> __It relies on generic and textual events__ </span>

<span style="color:#002060"> __and does not consider visually contextualized information\.__ </span>

<img src="img/VisualCOMET17.png" width=500px />

\[3\]Sap\, Maarten\, et al\. "Atomic: An atlas of machine commonsense for if\-then reasoning\." Proceedings of the AAAI Conference on Artificial Intelligence\. Vol\. 33\. No\. 01\. 2019\.

<span style="color:#00002F">Problem Statement</span>

How to close gap betweenperception\-levelandcognition\-level

visual understanding?

Theyintroduce a __new task__ of visual commonsense reasoning forcognitive visual scene understanding\, to reason about events before and after and people’s intents at present\.

They present the firstlarge\-scalerepository of Visual Commonsense Graphs that contains more than1Mtextual descriptions ofcommonsense inferencesover 60K complex visual scenes\.

Theyextend the GPT\-2model to incorporate visual information and allow direct supervision for grounding people in images\.

Cognitive image understanding via Visual Commonsense Graph

<img src="img/VisualCOMET18.png" width=380px />

<img src="img/VisualCOMET19.png" width=500px />

<span style="color:#002060"> __Definition of Visual Commonsense Graphs__ </span>

Cognitive image understanding via Visual Commonsense Graph

<img src="img/VisualCOMET20.png" width=500px />

__Inspired__ by designed of ATOMIC\, __if\-then reasoning__

but tailored specifically for __cognitive understanding of visual scene\.__

ATOMIC \(Sap\, M et al\. 2019\)

Definition of Visual Commonsense Graphs

<span style="color:#002060"> __Graph consists of four major components\!__ </span>

<img src="img/VisualCOMET21.png" width=500px />

<span style="color:#002060">A set of textual descriptions of</span>  <span style="color:#002060"> __events at present__ </span>

<span style="color:#002060">A set of commonsense inferences on</span>  <span style="color:#002060"> __events before__ </span>

<span style="color:#002060">A set of commonsense inferences on</span>  <span style="color:#002060"> __events after__ </span>

<span style="color:#002060">A set of commonsense inferences on people’s</span>  <span style="color:#002060"> __intents at present__ </span>

Definition of Tasks

<img src="img/VisualCOMET22.png" width=500px />

<img src="img/VisualCOMET23.png" width=500px />

<img src="img/VisualCOMET24.png" width=500px />

<span style="color:#002060"> __Generate the rest of visual commonsense graph__ </span>

<span style="color:#002060"> __that is connected to the specific current event__ </span>

Definition of Tasks

<img src="img/VisualCOMET25.png" width=500px />

<img src="img/VisualCOMET26.png" width=500px />

<img src="img/VisualCOMET27.png" width=500px />

<span style="color:#002060"> __Generate the complete set of__ </span>

<span style="color:#002060"> __commonsense inferences from scratch__ </span>

<span style="color:#00002F">Dataset Overview</span>

<span style="color:#002060">First</span>  <span style="color:#002060"> __large\-scale dataset__ </span>  <span style="color:#002060">of Visual Commonsense Graphs for images with</span>  <span style="color:#002060"> __person grounding__ </span>

<img src="img/VisualCOMET28.png" width=500px />

<img src="img/VisualCOMET29.png" width=500px />

<span style="color:#00002F">How to Generate?</span>

1\)  Make tags identifying each person in image

<img src="img/VisualCOMET30.png" width=324px />

<img src="img/VisualCOMET31.png" width=306px />

2\) Write event description

Source of images

Complex visual scenes with multiple people and activities

3\) Write intent inferences at the same time

__Stage 1: Grounded event descriptions with locations and intents__

<span style="color:#002060">※ For informative annotations\, description event with inferences \!</span>

<span style="color:#00002F">How to Generate?</span>

Originally\, part of movie scene

<img src="img/VisualCOMET32.png" width=500px />

1\) Showcrowdworkersfast\-forwarded clips of events

2\) Two workers for each event\, ask each to annotate

between two and four before and after inferences\.

__Stage 2: Collecting Before and After Inferences__

<img src="img/VisualCOMET33.png" width=495px />

<img src="img/VisualCOMET34.png" width=500px />

<img src="img/VisualCOMET35.png" width=441px />

<img src="img/VisualCOMET36.png" width=418px />

<img src="img/VisualCOMET37.png" width=110px />

<img src="img/VisualCOMET38.png" width=277px />

<img src="img/VisualCOMET39.png" width=495px />

Architecture : Pre\-trained GPT\-2

Single Stream Vision\-Language Transformer

<img src="img/VisualCOMET40.png" width=500px />

<img src="img/VisualCOMET41.png" width=441px />

<img src="img/VisualCOMET42.png" width=418px />

<img src="img/VisualCOMET43.png" width=110px />

<img src="img/VisualCOMET44.png" width=277px />

__PG trick : person tag token embedding add to__  __RoI__  __features__

__Sequence of inputs uses special tokens__

__Inference types start token to generate__

Objective function

<img src="img/VisualCOMET45.png" width=495px />

<img src="img/VisualCOMET46.png" width=500px />

<img src="img/VisualCOMET47.png" width=441px />

<img src="img/VisualCOMET48.png" width=418px />

<img src="img/VisualCOMET49.png" width=110px />

<img src="img/VisualCOMET50.png" width=277px />

__ They add EP loss for general version of model__

__\[Datasets\]__

https://visualcomet\.xyz/dataset

Publicly available

Publicly available

https://github\.com/jamespark3922/visual\-comet

Experiments setup

<img src="img/VisualCOMET51.png" width=485px />

Experiment setup

automatic metric for MT evaluation

Automatic metrics used in image captioning

Inference sentences that are unique

Total number of sentences

Unique : the number of inference sentences that are unique

within the generated sentences\.

Novel : the number of Generated sentences that are not in training data\.

Generated sentences that are not in training data

Total number of sentences

\[4\]Vedantam\, R\.\, LawrenceZitnick\, C\.\, & Parikh\, D\. \(2015\)\. Cider: Consensus\-based image description evaluation\. In Proceedings of the IEEE conference on computer vision and pattern recognition \(pp\. 4566\-4575\)\.

\[5\] Banerjee\,Satanjeev\, and AlonLavie\. "METEOR: An automatic metric for MT evaluation with improved correlation with human judgments\." Proceedings of theaclworkshop on intrinsic and extrinsic evaluation measures for machine translation and/or summarization\. 2005\.

Experiment setup

Automatic Evaluation

BLEU\-2

METEOR

CIDER

Quality of inference sentences

Acc@50

Unique

Novel

Filter out inference that do not match the content in image and event at present

Diversity of sentences

<img src="img/VisualCOMET52.png" width=500px />

<img src="img/VisualCOMET53.png" width=500px />

__Visual and textual__

__Modalities outperform__ models trained one modality

<img src="img/VisualCOMET54.png" width=500px />

__0 \-\-\-\-\-\-\-\-\-\-\-\-\-\- 1__

__likely               unlikely__

Human Evaluation

<img src="img/VisualCOMET55.png" width=500px />

<img src="img/VisualCOMET56.png" width=500px />

 __Text only model performs better than the Image \+    	Text model without text input in test time\.__

Qualitative Examples

<img src="img/VisualCOMET57.png" width=500px />

<img src="img/VisualCOMET58.png" width=500px />

<img src="img/VisualCOMET59.png" width=500px />

 __They state visual and text modalities generate more consistent and better contextualized predictions\.__

<img src="img/VisualCOMET60.png" width=500px />

Qualitative Examples

<img src="img/VisualCOMET61.png" width=500px />

<img src="img/VisualCOMET62.png" width=500px />

<img src="img/VisualCOMET63.png" width=500px />

 __They state visual and text modalities generate more consistent and better contextualized predictions\.__

They present Visual Commonsense Graphs\,the first large\-scalerepository of visual commonsense with more than1\.4 millioncommonsenseinferencesover60k imageswith complex visual scenes\.

Promising results that use the power of pre\-trained language to close gap betweenperception\-levelandcognition\-levelvisual understanding\.

The dataset supports new tasks and models beyond what are presented\.

Experiment setup

Baselines based on Different Inputs

\- Same architecture but ablate on the inputs \(place\, event\, and image\)

* Difference between inference and captioning
  * \(a\) generates sentences that are more diverse and rich in content than the captioning models \(b\)\.

<img src="img/VisualCOMET64.png" width=500px />

<img src="img/VisualCOMET65.png" width=500px />

