#version 450
layout(location=0) in vec4 markerColor;
layout(location=0) out vec4 outColor;
// Кубы и конусы имеют постоянный цвет: освещение на них не рассчитывается.
void main() { outColor = markerColor; }
