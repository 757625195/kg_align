from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "scripts" / "build_current_springer_english_manuscript.py"
OUTPUT = ROOT / "outputs" / (
    "Cross_Lingual_Knowledge_Graph_Entity_Alignment_"
    "Springer_English_Clear_Style_20260830.docx"
)


def load_builder():
    spec = importlib.util.spec_from_file_location("springer_builder", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OVERRIDES = {
    4: (
        "Cross-lingual knowledge graph entity alignment identifies entities that refer to the same real-world object in graphs written in different languages. "
        "Existing methods use graph structure and text. However, they often model relation types, propagation depths, and short text pieces separately. "
        "They also provide limited evidence about the role of structural neighbors in a joint representation. This study combines relation-aware structural context with multi-scale semantics. "
        "The structural encoder uses shared relation projections and selects useful propagation depths for each node. The semantic encoder builds token, phrase, and global views from the same entity text. "
        "The fusion module uses 1.5-entmax to select evidence from the complete one-hop outgoing neighborhood. It then combines this evidence with the semantic representation. "
        "The model uses bidirectional InfoNCE for training and CSLS for retrieval. The experiments cover five datasets and three random seeds. Hits@1 ranges from 0.6344 to 0.9239. "
        "The largest sample standard deviation is 0.0053. Removing relation types lowers Hits@1 by 3.62 points on DBP15K ZH-EN and 2.78 points on OpenEA EN-FR-15K-V2. "
        "Semantic-only retrieval lowers it by 9.09 and 9.66 points. Under the same neighborhood and model size, structural queries are 2.08 and 2.48 points below semantic queries. "
        "Late fusion without neighbor context is 5.28 and 9.22 points below the full model. These results support the use of relation-aware neighbor evidence. "
        "However, the late-fusion gap does not measure the effect of fusion timing alone."
    ),
    8: (
        "Knowledge graphs usually represent facts as triples {{EQ0}}. In a triple, {{EQ1}}, {{EQ2}}, and {{EQ3}} denote the head entity, relation, and tail entity. "
        "The triple states that relation {{EQ5}} links head entity {{EQ4}} to tail entity {{EQ6}} [1]. Different institutions and information systems often build knowledge graphs independently. "
        "They may also use different languages. As a result, the same real-world object may have different identifiers, names, attributes, and value formats. "
        "Entity alignment finds matching entities in sets {{EQ7}} and {{EQ8}}. These matches support graph integration."
    ),
    9: (
        "Existing methods follow three main lines. MTransE and JAPE map different graphs into shared or linked vector spaces [2, 3]. "
        "They place matching entities close together. BootEA repeatedly selects high-confidence pairs and uses them as pseudo-labels [4]. "
        "RDGCN and RREA use relations and neighborhood structure to improve structural representations [5, 6]. These studies provide the main basis for cross-lingual entity alignment. "
        "However, a unified model still needs to combine structural and textual evidence more clearly."
    ),
    10: (
        "First, similar graph structure does not always mean that two entities are equal. Person A may connect to Beijing through a birthplace relation. "
        "Person B may connect to Beijing through a workplace relation. An encoder may overestimate their similarity if it treats both relations as the same edge type [5, 7]. "
        "Second, text evidence is often incomplete. The token 'Washington' cannot show whether an entity is a person, a state, or a city. "
        "The model needs relation or attribute context. Third, the available seed alignments usually cover only part of the entity sets [4, 8]. "
        "The model must therefore learn useful structural and semantic patterns from a small set of known pairs."
    ),
    12: (
        "This study asks three questions. First, does the model produce accurate and stable results on five datasets? "
        "Second, how much do the structural encoder, semantic encoder, and neighbor aggregation each contribute? "
        "Third, how do semantic and structural neighbor queries compare under the same neighborhood and parameter count? "
        "The study also measures the change caused by removing neighbor context from a late-fusion baseline."
    ),
    13: "This study makes three contributions.",
    14: (
        "This study develops a relation-aware structure-semantic model for entity alignment. The structural encoder keeps edge types and combines several propagation depths. "
        "The semantic encoder extracts word, local phrase, and global context from the same entity text. A shared alignment objective trains both representations."
    ),
    15: (
        "The model uses the complete one-hop structural neighborhood before it builds the joint representation. The model gives each neighbor a sparse relevance weight. "
        "It then combines the neighbor summary, the entity's own structure, and its semantics. Two controls help explain the result. "
        "One control uses a structural query. The other uses late fusion with the same number of trainable parameters."
    ),
    16: (
        "The experiments use three independent runs on five entity alignment datasets. Tests on two main datasets measure the effects of the structural, semantic, and neighbor components. "
        "The tests also show the limits of the current evidence."
    ),
    19: (
        "MTransE represents entities and relations as vectors in a separate space for each language [2]. It learns mappings between these spaces and brings matching entities closer together. "
        "JAPE adds attribute links to a shared structural space [3]. It lets names, attributes, and graph structure support the same match. JAPE also introduced the widely used DBP15K dataset. "
        "However, vector translations and linear mappings have limits. They may not describe one-to-many relations or complex neighborhoods well. They may also change distances in the original spaces."
    ),
    21: (
        "GraphSAGE learns node representations by sampling and combining local neighbors [9]. R-GCN also keeps relation types during message transformation [7]. "
        "It uses basis or block decomposition to limit the number of parameters. These studies show that neighbor aggregation should retain both entities and edge types. "
        "Relation frequency differs across graphs. Different nodes may also need different propagation depths. A model must handle both issues without using too many parameters."
    ),
    22: (
        "RDGCN adds relation evidence through attention between an entity graph and a relation dual graph [5]. RREA uses relation reflection to keep relation differences and space geometry [6]. "
        "RPR-RHGT generates and filters multi-step relation paths [10]. It then encodes these paths with a relation-aware graph Transformer. "
        "These studies show that relation types and multi-hop structure can improve entity alignment."
    ),
    24: (
        "Transformers use multi-head self-attention to connect different positions in a sequence [11]. Hierarchical attention networks show that different levels of text may contribute in different ways [12]. "
        "These results support the use of word cues, local phrases, and global context. However, they do not prove that every semantic view helps entity alignment. "
        "The present model uses all three views in one encoder. Separate ablation tests measure the value of each view."
    ),
    25: (
        "Text-based entity alignment studies show that names, descriptions, and other data types can support graph structure. RREA reports text-based and structure-only settings separately [6]. "
        "MCLEA learns several data types with contrastive and cross-modal objectives [13]. Another knowledge graph Transformer study finds that simple concatenation or attention may mix spaces that are not aligned [14]. "
        "These studies support structure-semantic fusion. However, they change both the model design and the training objective. Their results cannot show the effect of interaction position alone."
    ),
    26: (
        "Recent studies also use large language models under noisy labels [15]. Other studies use multi-source domain adaptation for real-world knowledge graphs [16]. "
        "Entity alignment agents also use step-by-step reasoning [17]. These methods change either the supervision source or the reasoning process."
    ),
    29: "We define the two knowledge graphs as follows.",
    31: (
        "In this definition, {{EQ0}}, {{EQ1}}, and {{EQ2}} are the entity, relation, and triple sets on side {{EQ3}}. "
        "The task also provides a small seed alignment set."
    ),
    33: (
        "The set {{EQ0}} contains matching entity pairs that human labels or dataset rules confirm. The symbol {{EQ1}} states that the two entities refer to the same real-world object."
    ),
    34: (
        "The model learns a scoring function {{EQ0}}. This function should rank the true match above all other candidates. The evaluation uses a one-to-one, closed-world setting."
    ),
    36: (
        "The model has four parts. They are a relation-aware structural encoder, a multi-scale semantic encoder, a neighbor fusion module, and a contrastive learning objective. "
        "The structural encoder passes relation-aware messages on both graphs. It also combines several propagation depths. "
        "The semantic encoder builds token, phrase, and global views from names, relations, and attributes. The fusion module selects useful evidence from the complete one-hop outgoing neighborhood. "
        "It combines this evidence with the semantic representation. The training objective acts on the joint and structural representations. "
        "The retrieval stage uses CSLS for cross-graph matching. Figure 1 links each step to its equations."
    ),
    38: (
        "Fig. 1 Relation-aware structural context and multi-scale semantic fusion. The structural branch passes messages along incoming edges and selects useful propagation depths. "
        "The semantic branch builds token, phrase, and global views from one input. The fusion module reads all one-hop outgoing neighbors in the original edge direction. "
        "It assigns sparse weights with 1.5-entmax. The model then combines the structural context with the semantic representation. "
        "The training and retrieval stages both use the joint representation. Validation selects only {{EQ0}}."
    ),
    40: (
        "Local degree and neighbor statistics often describe a node's structural role [18, 19]. Structural-identity methods also compare nodes by their role in the graph. "
        "This study therefore uses local topology features to start the structural branch. This study defines the exact feature set below."
    ),
    41: (
        "The local topology vector has eight values. The model computes this vector separately in each graph. The computation does not require matching relation identifiers across graphs. "
        "For entity {{EQ0}}, the variables {{EQ1}}, {{EQ2}}, and {{EQ3}} represent in-degree, out-degree, and total degree. "
        "The variables {{EQ4}} and {{EQ5}} count distinct incoming and outgoing relation types. The variables {{EQ6}} and {{EQ7}} give the mean {{EQ8}} value for incoming source entities and outgoing target entities. "
        "The model uses zero when a node has no such neighbor. The variable {{EQ9}} measures the balance between incoming and outgoing edges. "
        "The model defines the raw feature vector as follows."
    ),
    43: (
        "The model standardizes each topology feature within the left and right graph. This step produces {{EQ0}}. The initial structural state uses a shared topology projection and a small entity-specific residual."
    ),
    45: (
        "The matrix {{EQ0}} is a linear projection that both graphs share. The vector {{EQ1}} is a trainable entity vector. "
        "The topology term gives similar local roles a comparable starting point. The residual has a scale of 0.1. "
        "It keeps different entities with similar local statistics from receiving the same representation."
    ),
    46: (
        "During each forward pass, the structural encoder sends directed, relation-aware messages over the full graph. For triple {{EQ0}}, head entity {{EQ1}} sends a message to tail entity {{EQ2}}. "
        "The message follows the original edge direction. The encoder uses incoming edges only. The variable {{EQ3}} is the state of entity {{EQ4}} at layer {{EQ5}}. "
        "The variable {{EQ6}} is the relation embedding. The set {{EQ7}} contains the incoming edges of entity {{EQ8}}. Layer {{EQ9}} first computes the following message."
    ),
    48: (
        "The matrices {{EQ0}} and {{EQ1}} project the source entity and the relation embedding. All relations in one layer share these matrices. "
        "The embedding {{EQ2}} keeps the difference between relation types. This design avoids a separate full matrix for every relation. "
        "The vector {{EQ3}} is the message that edge {{EQ4}} sends to target entity {{EQ5}}. The model averages all messages that reach entity {{EQ6}}."
    ),
    50: (
        "The model first projects the entity's own state as {{EQ0}}. It then uses the self-state and the mean neighbor message to compute a gate for each dimension."
    ),
    53: (
        "In Eq. (3), {{EQ0}} gives each representation dimension a value from zero to one. Equation (4) keeps the projected self-state and adds neighbor evidence through this gate. "
        "The triple 'Yao Ming - birthplace -> Shanghai' gives a simple example. The message for Shanghai includes Yao Ming's state and the birthplace relation embedding. "
        "The encoder can therefore distinguish this edge from a workplace edge that also points to Shanghai."
    ),
    54: "Equation (4) gives the raw output {{EQ1}} of layer {{EQ0}}. The model transforms this output before the next layer, except after the final layer.",
    56: (
        "The model applies layer normalization, a nonlinear function, and dropout after each propagation layer. These steps keep feature values stable across depths and reduce overfitting."
    ),
    57: (
        "Stacked propagation lets each entity collect evidence from more distant neighbors. The path 'Yao Ming - birthplace -> Shanghai - country -> China' gives an example. "
        "In this path, evidence about Yao Ming can reach China through Shanghai."
    ),
    58: (
        "The deepest layer may lose information about the entity itself or its direct neighbors. The encoder therefore keeps the projected input and every raw layer output, {{EQ0}}. "
        "The state {{EQ1}} has no neighbor information. The state {{EQ2}} contains information from at most {{EQ3}} hops. "
        "The model first computes the mean context {{EQ4}}. A two-layer MLP then scores each state together with this mean context. "
        "A {{EQ5}} operation changes these scores into depth weights {{EQ6}}."
    ),
    61: (
        "The layer-selection MLP has input size {{EQ0}} and hidden size {{EQ1}}. It uses GELU and dropout and returns one score. "
        "Equation (7) gives a separate weight to each depth from {{EQ2}} to {{EQ3}}. LayerNorm and L2 normalization then produce structural representation {{EQ4}}. "
        "Each entity can therefore use a different mix of propagation depths."
    ),
    63: (
        "The three semantic views use the same entity text. Each view reads a different range of this text. "
        "The token view uses word-level clues. The phrase view uses nearby word groups. The global view connects positions across the full sequence."
    ),
    64: (
        "For entity {{EQ0}}, the model joins its name, nearby relation names, and attribute text into one token sequence. The model maps each token to a word vector. "
        "The phrase and global views use this same sequence. They do not read extra descriptions, outside sentences, or other text sources. "
        "A shared projection, normalization step, and position encoding produce base representation {{EQ1}}. Mask {{EQ2}} marks the valid tokens."
    ),
    65: (
        "The Yao Ming sequence gives a simple example. It may contain the name 'Yao Ming', the relations 'birthplace' and 'team', and the attribute 'height 2.29 m'. "
        "The token view uses single clues such as 'Yao' and 'height'. The phrase view uses nearby groups such as 'birthplace Shanghai' and 'team Rockets'. "
        "The global view connects the name, relations, and distant attributes across the full sequence. The views use different text ranges, but they use the same source."
    ),
    66: (
        "The token view summarizes the base sequence directly. Mean pooling keeps information from every valid token. Attention pooling gives more weight to useful tokens. "
        "The model averages the two outputs to form the word-level representation."
    ),
    68: (
        "The model normalizes attention weights over valid positions only. Padding tokens therefore do not affect the entity representation. "
        "Mean pooling keeps broad word evidence. Attention pooling stops a few high-weight tokens from fully controlling the representation."
    ),
    69: (
        "The phrase view combines nearby tokens in the base sequence. This step captures local patterns in names, relation phrases, and attribute text."
    ),
    72: (
        "The global view uses a Transformer encoder to connect distant tokens. This step lets name, relation, and attribute clues affect one another across the full entity text."
    ),
    75: (
        "Different entities may rely on different levels of text. The model therefore estimates a weight for each view from the three view representations."
    ),
    77: (
        "The model weights the three view vectors and joins them. An output MLP processes the joined vector. "
        "The model adds this output to a projected sum of the weighted views. L2 normalization produces semantic representation {{EQ0}}."
    ),
    79: (
        "This component selects structural evidence from the complete one-hop outgoing neighborhood. It then combines this evidence with the entity's semantic representation. "
        "The computation has three stages. First, a semantic query scores each structural neighbor. The 1.5-entmax function gives an exact zero weight to weak candidates. "
        "Second, the model combines the weighted neighbor summary with the entity's own structure. This step forms the structural context. "
        "Third, a gate combines the structural context with the semantic representation. This step produces the joint representation."
    ),
    81: (
        "For entity {{EQ0}}, the fusion module receives four inputs. They are semantic representation {{EQ1}}, self structural representation {{EQ2}}, all one-hop outgoing neighbor states, and a validity mask. "
        "A projection changes the semantic representation into query {{EQ3}}. The model projects structural neighbor {{EQ4}} into key {{EQ5}} and value {{EQ6}}. "
        "Their scaled dot product gives the first relevance score."
    ),
    83: (
        "Equation (14) measures the match between the entity semantics and structural neighbor {{EQ0}}. A larger dot product means a stronger semantic match. "
        "The scale factor keeps high-dimensional dot products in a stable range. The variable {{EQ1}} is the size of the query and key vectors."
    ),
    84: (
        "A dot product may miss detailed relations between matching vector dimensions. The model therefore defines the following interaction feature."
    ),
    86: (
        "The part {{EQ0}} keeps both input vectors. The part {{EQ1}} shows which dimensions are active in both vectors. "
        "The part {{EQ2}} shows the absolute difference in each dimension. A learned mapping and a sigmoid function use these features to produce a neighbor gate. "
        "Natural language inference models use the same basic feature pattern [20]. The pattern is not the final similarity score. "
        "It gives the model information for judging whether a neighbor fits the current entity."
    ),
    88: (
        "The gate value {{EQ0}} shows how well neighbor {{EQ1}} passes this feature check. The model uses the dot-product score and gate value to compute normalized neighbor weights."
    ),
    90: (
        "In Eq. (17), {{EQ0}} contains the scores of all valid neighbors. The vector {{EQ1}} contains the gate values. Mask value {{EQ2}} removes padded positions. "
        "Softmax gives every valid position a positive weight. In contrast, 1.5-entmax [21] can give weak neighbors an exact zero weight. "
        "Temperature {{EQ3}} controls how focused the weights are. The valid weights still sum to one. An entity without valid neighbors receives a zero summary."
    ),
    91: (
        "The Yao Ming example also explains neighbor selection. Its outgoing neighbors may include the Houston Rockets, Shanghai, and an unrelated entity. "
        "The clues 'basketball player' and 'team' may raise the score of the Rockets. The 1.5-entmax function can give the unrelated entity a zero weight. "
        "Equation (19) then decides how much of the entity's own structure and neighbor evidence the model should keep."
    ),
    93: (
        "The model uses the sparse weights from Eq. (17) to average the neighbor value vectors. This step produces structural neighbor summary {{EQ1}} for entity {{EQ0}}. "
        "The full model keeps every one-hop outgoing neighbor before weighting. The summary can therefore use the full local neighborhood. "
        "Sparse weights reduce the effect of unrelated neighbors."
    ),
    94: (
        "The model then uses the self structural representation, neighbor summary, and semantic representation to compute a context gate for each dimension."
    ),
    97: (
        "Gate vector {{EQ0}} controls the balance between the entity's own structure and the neighbor summary in each dimension. A value near one makes Eq. (19) favor {{EQ1}}. "
        "A value near zero makes it favor {{EQ2}}. If the entity has no valid neighbor, the model sets {{EQ3}}. "
        "This rule prevents padding from changing the entity representation."
    ),
    99: (
        "After the model obtains structural context {{EQ0}}, it computes a joint gate. This gate sets the share of structural and semantic evidence in each dimension."
    ),
    102: (
        "Equation (21) first produces gated vector {{EQ0}}. A joint-gate value near one gives more weight to structural context. "
        "A value near zero gives more weight to semantic evidence. The full model then computes the equal-weight base vector {{EQ1}} and output {{EQ2}}. "
        "This fixed residual limits large changes from the gate. It does not add trainable parameters."
    ),
    103: (
        "The semantic query changes neighbor weights before the model builds the neighbor summary. The model then combines the new structural context with the semantic representation. "
        "This study calls the process semantic neighbor querying. However, the model design alone cannot prove that this query is useful. "
        "Section 3.5.4 therefore introduces a structural-query control with the same neighborhood. In this control, semantics enters only at the final joint gate."
    ),
    105: (
        "Two controls test the query signal and the removal of neighbor context. The structural-query control keeps the same neighborhood, dynamic padding, 1.5-entmax, and gates. "
        "It uses the entity's structural representation to compute neighbor weights. Semantics enters only after the model forms the structural context. "
        "The late-fusion control has the same number of trainable parameters as the full model. It does not use neighbor context. "
        "It builds a training representation after the structural and semantic branches finish their separate encoding steps."
    ),
    107: (
        "Equation (22) normalizes the structural and semantic representations separately. It then joins them and applies a nonlinear mapping. "
        "This process produces {{EQ0}} for InfoNCE training and retrieval. The baseline has the same number of trainable fusion parameters as the full model."
    ),
    109: (
        "The model uses single-stage bidirectional InfoNCE training [22]. It does not use a separate warm-up stage. Each mini-batch samples {{EQ0}} pairs from the seed alignment set. "
        "The same pairs produce the joint and structural losses. The model updates both losses in one forward and backward pass. "
        "The following equation gives the temperature-scaled cosine similarity between left entity {{EQ1}} and right entity {{EQ2}}."
    ),
    111: (
        "The variable {{EQ0}} is the temperature. Correct pairs appear on the diagonal of the similarity matrix. "
        "The model computes cross-entropy in both retrieval directions. It averages the two values to obtain joint loss {{EQ1}}."
    ),
    113: (
        "Equation (24) directly trains the joint representation. The topology-based structural branch also needs a shared space across the two graphs. "
        "The model therefore applies the same bidirectional InfoNCE form to {{EQ0}} early in training. This step gives structural loss {{EQ1}}. "
        "The following equation gives the total objective."
    ),
    115: (
        "The model uses structural supervision only in the early part of training. Its weight falls in a straight line as training continues. "
        "The variable {{EQ0}} is the current epoch, and {{EQ1}} is the total number of epochs. The model defines training progress and structural weight as follows."
    ),
    117: (
        "The structural loss has a weight of 0.1 in the first epoch. Its weight reaches zero in the final epoch."
    ),
    119: (
        "Equation (25) trains the encoders. The main retrieval step uses the trained joint representation {{EQ0}} directly. "
        "It does not use validation data to change the structural or semantic weight. CSLS [23] corrects cosine similarities between joint representations."
    ),
    121: (
        "The terms {{EQ0}} and {{EQ1}} are mean cosine similarities. They use the {{EQ2}} nearest entities on the other graph for the query and candidate. "
        "For each run, validation MRR selects {{EQ3}} from the candidate set. The test set is then evaluated once."
    ),
    126: (
        "The evaluation ranks each entity against the full candidate set. It reports Hits@1, Hits@10, and mean reciprocal rank (MRR). "
        "The main and ablation results show the mean and sample standard deviation over three random seeds."
    ),
    128: (
        "The model uses AdamW [24], a batch size of 512, and 128-dimensional representations. The structural encoder has three relation-aware GNN layers. "
        "All components use a dropout rate of 0.1. The model projects each 300-dimensional semantic input to 128 dimensions. "
        "The global view uses a two-layer Transformer with four attention heads. The phrase view uses convolution widths of 3 and 5. "
        "The 1.5-entmax temperature is {{EQ0}}, and the InfoNCE temperature is {{EQ1}}. The experiments use random seeds 42, 43, and 44. "
        "DBP15K training lasts at most 36 epochs. OpenEA and EventEA training lasts at most 50 epochs. "
        "The model checks validation MRR every five epochs. Training stops after four checks without improvement. "
        "The main retrieval step does not tune a weight between the two branches. Instead, validation MRR selects one {{EQ2}} value for each run. "
        "Preprocessing matches relation identifiers after it normalizes relation names. DBP15K also uses sup_rel_ids to map relation pairs that the dataset provides."
    ),
    131: (
        "Table 2 reports test results from the joint representation on five datasets. The model uses three random seeds for each dataset. "
        "DBP15K reserves 10% of its training pairs for validation. OpenEA and EventEA use their official validation sets."
    ),
    134: (
        "Mean Hits@1 ranges from 0.6344 to 0.9239 across the five datasets. The largest sample standard deviation across the three seeds is 0.0053."
    ),
    136: (
        "Table 3 lists results reported in earlier studies. The DBP15K results for MTransE, JAPE, BootEA, and RDGCN come from the RDGCN paper [5]. "
        "The RREA-text result comes from the text-based setting in the RREA paper [6]. The OpenEA EN-FR-V2 results are the official five-fold averages from OpenEA [8]."
    ),
    139: (
        "The methods in Table 3 use different data splits, text inputs, and post-processing steps. The table therefore provides context rather than a strict ranking under one protocol. "
        "On DBP15K ZH-EN, the proposed model performs better than the earlier structural methods in the table. It performs worse than RREA-text. "
        "On OpenEA EN-FR-V2, it performs worse than BootEA, KDCoE, and RDGCN. These results do not support a state-of-the-art claim."
    ),
    142: (
        "Table 4 reports ablation tests on DBP15K ZH-EN and OpenEA EN-FR-15K-V2. Each variant uses three random seeds. "
        "Each variant also uses the same representation for training and retrieval."
    ),
    144: (
        "Table 4 shows several effects. Learnable entity vectors without topology features lower Hits@1 by 3.46 points on DBP15K and 8.03 points on OpenEA. "
        "Removing relation types lowers Hits@1 by 3.62 and 2.78 points. Removing the layer selector causes a smaller drop on both datasets. "
        "The fixed eight-neighbor plus softmax control also causes a smaller drop. The phrase view helps on both datasets. "
        "The token view changes Hits@1 by no more than 0.27 points. Removing the global view raises DBP15K by 0.49 points but lowers OpenEA by 1.15 points. "
        "The global view therefore has a dataset-specific effect. Removing structural supervision lowers Hits@1 by only 0.07 and 0.17 points. "
        "The structural loss is therefore not a main source of the result."
    ),
    146: (
        "Table 5 separates the training and retrieval representations. It includes the full model, the two single branches, and mean fusion without trainable parameters. "
        "Each result uses the same representation for training and retrieval. Validation does not change the structural or semantic weight."
    ),
    148: (
        "The semantic-only branch is much stronger than the structural-only branch. However, it remains below the full model. "
        "Its Hits@1 is 9.09 points lower on DBP15K and 9.66 points lower on OpenEA. Mean fusion without trainable parameters also remains below the full model. "
        "These results show that structural information helps most when the model learns it together with semantic evidence. Structural information is less useful when retrieval uses it alone."
    ),
    150: (
        "Table 6 compares three models. The full model and structural-query control use the same neighborhood, fusion parameters, and joint retrieval. "
        "Only the signal for neighbor scoring changes. The late-fusion model has the same number of trainable fusion parameters. However, it does not read neighbor context."
    ),
    152: (
        "The comparison holds the neighborhood, training capacity, and retrieval representation constant. The results show that structural queries yield Hits@1 values 2.08 and 2.48 points below semantic queries. "
        "Both datasets show the same direction. Semantic querying therefore helps under the current setting. Late fusion is 5.28 and 9.22 points below the full model. "
        "This control removes neighbor context and changes the training interaction path. Its gap cannot measure the effect of fusion timing alone."
    ),
    155: (
        "The experiments give three main findings. First, results vary little across random seeds, although the five datasets have different performance levels. "
        "Second, the semantic-only branch is much stronger than the structural-only branch. However, the full model performs better than both single branches and mean fusion. "
        "The topology and relation ablations also support this result. Structural evidence helps most inside a learned joint representation. "
        "Third, semantic queries perform better than structural queries when the two controls use the same neighborhood and model size. "
        "Late fusion without neighbor context performs worse. The query signal and neighbor context therefore both affect the joint representation. "
        "The semantic views do not help every dataset in the same way. The global view shows the clearest dataset-specific effect."
    ),
    156: (
        "DBP15K provides useful names and local graph clues. EventEA combines Wikidata and DBpedia. It contains more varied event relations and attributes [25]. "
        "OpenEA EN-FR-15K-V2 encodes entity URIs to reduce name bias [8]. The current input uses English GloVe vectors for known tokens [26]. "
        "It uses fixed random vectors for unknown tokens. An MD5 seed makes these random vectors repeatable. GloVe covers about 52.18% of the OpenEA tokens. "
        "Many proper names and encoded identifiers therefore lack useful cross-lingual meaning. This pattern agrees with the lower OpenEA result. "
        "However, the current experiments do not prove that low word coverage causes the gap. A future test must replace the semantic encoder while keeping input coverage controlled."
    ),
    158: (
        "The study has several limits. The main and ablation tests use the same splits, validation rule, and three random seeds. "
        "However, {{EQ0}} is too small for strong significance tests. The fixed eight-neighbor plus softmax control changes both the neighbor count and the weight function. "
        "The late-fusion control changes both neighbor visibility and the training interaction path. Neither control isolates one cause. "
        "The experiments also do not test relation-identifier sharing and edge direction separately. The component tests cover only two main datasets. "
        "The OpenEA result uses one official split. The study also does not test open-world data with unmatched entities."
    ),
    160: (
        "Future work should focus on three tasks. First, cross-lingual subword or multilingual retrieval encoders should replace random vectors for unknown tokens [27, 28]. "
        "The new model should then be tested on all official OpenEA splits. Second, larger graphs should test the speed and stability of the structural encoder and sparse neighbor selection. "
        "Third, open-world tests should include entities without matches. These tests should also study retrieval score adjustment across datasets."
    ),
    162: (
        "This study presents a joint model of relation-aware structural context and multi-scale semantics. It tests the model with several random seeds on five datasets. "
        "Topology features, relation types, and the phrase view add useful information. The structural branch is weak when the model uses it alone. "
        "However, the full model performs better than the semantic-only branch and mean fusion. Semantic queries also perform better than structural queries on two datasets with the same neighborhood. "
        "The late-fusion control gives lower results. However, that control changes both neighbor visibility and the training interaction path. "
        "The results support relation-aware neighbor context and semantic neighbor queries in the two controlled datasets. More experiments must test whether these findings hold on other datasets. "
        "Future work must also test the value of the structural loss on its own."
    ),
    164: (
        "Data availability  DBP15K, OpenEA, and EventEA are public datasets. The reference list gives their sources."
    ),
    165: (
        "Code availability  The corresponding author can provide the code, settings, and result summaries upon reasonable request."
    ),
}


def main() -> None:
    builder = load_builder()
    builder.P.update(OVERRIDES)
    builder.OUTPUT = OUTPUT
    builder.main()

    document = builder.Document(OUTPUT)
    replacements = {
        "Funding": (
            "Funding  The author must add the correct funding statement before submission. "
            "If the study received no external funding, the author may use this statement: "
            "'The author received no specific funding for this work.'"
        ),
        "Author contributions": (
            "Author contributions  Xinyi Lin designed the study and developed the method and software. "
            "Xinyi Lin ran and analyzed the experiments. Xinyi Lin also prepared the data and figures and wrote and revised the manuscript."
        ),
        "Ethics approval": "Ethics approval  This study did not require ethics approval.",
        "Consent to participate": "Consent to participate  This study did not involve human participants.",
        "Consent for publication": "Consent for publication  This study did not require consent for publication.",
    }
    for paragraph in document.paragraphs:
        current = builder.combined_text(paragraph).strip()
        for prefix, replacement in replacements.items():
            if current.startswith(prefix):
                builder.set_plain_text(paragraph, replacement)
                break
    document.save(OUTPUT)


if __name__ == "__main__":
    main()
