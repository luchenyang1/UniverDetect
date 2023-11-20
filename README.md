# UniverDetect
Landmark detection using X-ray imaging is vital for disease screening, treatment, and prognosis. It provides a framework for subsequent tasks, including segmentation, classification, and target detection. Existing landmark detection methods exhibit limits with the increasing number of X-ray examiners. Although they serve specific data types,  their ability to learn the complete semantic information from this dataset is limited, potentially affecting their widespread adoption. We address these challenges by proposing a universal landmark detection model for multidomain X-ray imaging, UniverDetect. By progressively acquiring local semantic information, global semantic insights, and anatomical structure knowledge at various levels, the model ensured that the detected landmarks were closely aligned with the ground-truth labels. UniverDetect consists of three key components. The landmark detection module (LDM) utilizes a U-net network equipped with our innovative pyramid depthwise separable (PDS) convolution module for initial landmark detection. The landmark refinement module (LRM) integrates a fine-tuning module comprising continuously extended convolution blocks. Finally, the landmark correction module (LCM) incorporates a graph convolutional network (GCN) to rectify offset errors during partial landmark detection. An inherent feature of this model is its domain- generalization capability, which enables continuous learning across diverse domains. This model can concurrently  learn from eight domains, covering 118 landmarks within a diverse dataset of 5969 images. Extensive experiments on multiple datasets demonstrate that this method consistently outperforms state-of-the-art approaches. This study presents a versatile and effective solution to reduce doctors’ workload while providing precise quantitative analysis. 
