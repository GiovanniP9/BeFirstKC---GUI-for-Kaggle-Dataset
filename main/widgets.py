# widgets.py
from PyQt5 import QtWidgets, QtCore, QtGui

class AnimatedButton(QtWidgets.QPushButton):
    """
    QPushButton custom con effetto glow animato al passaggio del mouse.
    """
    def __init__(self, *args, glow_color=QtGui.QColor("#F55555"), max_blur=20, **kwargs):
        super().__init__(*args, **kwargs)
        # Effetto glow esterno
        self._effect = QtWidgets.QGraphicsDropShadowEffect(self)
        self._effect.setOffset(0, 0)
        self._effect.setColor(glow_color)
        self._effect.setBlurRadius(0)
        self.setGraphicsEffect(self._effect)

        # Animazione di entrata
        self._enter_anim = QtCore.QPropertyAnimation(self._effect, b"blurRadius", self)
        self._enter_anim.setDuration(200)
        self._enter_anim.setStartValue(0)
        self._enter_anim.setEndValue(max_blur)

        # Animazione di uscita
        self._leave_anim = QtCore.QPropertyAnimation(self._effect, b"blurRadius", self)
        self._leave_anim.setDuration(200)
        self._leave_anim.setStartValue(max_blur)
        self._leave_anim.setEndValue(0)

        # Animazione colore glow
        self._color_anim = QtCore.QPropertyAnimation(self._effect, b"color", self)
        self._color_anim.setDuration(200)
        self._color_anim.setStartValue(glow_color.lighter(120))
        self._color_anim.setEndValue(glow_color)

    def enterEvent(self, event):
        """Avvia l'animazione di glow al passaggio del mouse."""
        self._enter_anim.start()
        self._color_anim.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Riduce il glow quando il mouse esce dal pulsante."""
        self._leave_anim.start()
        super().leaveEvent(event)
