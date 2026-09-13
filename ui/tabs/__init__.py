"""Cada pestana en su propio modulo. El orden es el de la interfaz."""
from ui.tabs.selector import modulo_selector_farmacia
from ui.tabs.configuracion import modulo_configuracion
from ui.tabs.business_intelligence import modulo_business_intelligence
from ui.tabs.pedidos import modulo_generador_pedidos
from ui.tabs.auditoria import modulo_auditoria
from ui.tabs.torre_control import modulo_torre_control
from ui.tabs.laboratorio_ml import modulo_laboratorio_ml

__all__ = [
    "modulo_selector_farmacia",
    "modulo_configuracion",
    "modulo_business_intelligence",
    "modulo_generador_pedidos",
    "modulo_auditoria",
    "modulo_torre_control",
    "modulo_laboratorio_ml",
]
