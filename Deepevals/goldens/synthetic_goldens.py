from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
from deepeval.models import OllamaModel
from deepeval.dataset import EvaluationDataset

PDF_PATH = "C:\\Users\\madhu\\FinancialChatbot-ChainBased\\testdata\\JPmorgan10kReport.pdf"

retrievermodel = OllamaModel(model="deepseek-r1:8b")

synthesizer = Synthesizer()

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=[PDF_PATH],
    include_expected_output=True,
    max_goldens_per_context=3,
    context_construction_config=ContextConstructionConfig(
        critic_model=retrievermodel,
        chunk_size=1200,
        chunk_overlap=500,
        max_retries=1
    )
)

dataset = EvaluationDataset(goldens=goldens)
dataset.push("Synthetic Golden from Doc")