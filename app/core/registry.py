from app.interfaces.eda import EDA
from app.interfaces.statistics import StatisticalAnalyser
from app.interfaces.model_manager import ModelManager
from app.interfaces.visualization import Visualization

# Mappatura dei nomi leggibili a classi concrete
CLASS_REGISTRY = {
    "EDA": EDA,
    "StatisticalAnalyser": StatisticalAnalyser,
    "ModelManager": ModelManager,
    "Visualization": Visualization
}
