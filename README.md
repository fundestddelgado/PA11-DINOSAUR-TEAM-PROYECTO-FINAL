# PA11-DINOSAUR-TEAM-PROYECTO-FINAL
REPOSITORIO PARA EL PROYECTO FINAL Y EL HACKATHON

INFORMACIÓN GENERAL DEL PROYECTO FINAL:

1. Planteamiento del Problema
    - El desafío principal abordado es la falta de autonomía e información contextual que enfrentan las personas con discapacidad visual en entornos sociales y domésticos. La identificación rápida y fiable de personas en su cercanía es crucial para la seguridad, la interacción social y la calidad de vida.
    - Las soluciones de reconocimiento facial convencionales suelen estar integradas en sistemas complejos o carecen de una interfaz de usuario accesible (como audio y una navegación sencilla).  La creación de un sistema de reconocimiento personal requiere un método eficiente para entrenar modelos con conjuntos de datos limitados (rostros familiares) y la capacidad de clasificar personas desconocidas.
    - La solución es desarrollar una herramienta de soporte visual y auditivo que utilice la visión por computadora para interpretar el entorno y comunique esta información de manera audible, eliminando la barrera de la visión.

2. Objetivos del Proyecto

    1. Identificar los rostros de personas con discapacidad visual en entornos sociales y domésticos.
    2. Crear un sistema de reconocimiento personal que reconozca rostros familiares y desconocidos.
    3. Desarrollar una herramienta de soporte visual y auditivo que utilice la visión por computadora para interpretar el entorno y comunique esta información de manera audible, eliminando la barrera de la visión.

3. Herramientas utilizadas en el proyecto

    1. Python
    2. Visión por Computadora (IA): TensorFlow / Keras como marco principal para construir, entrenar y cargar la Red Neuronal Convolucional (CNN) que clasifica los rostros.
    3. Sistema de Detección de Rostros: MTCNN como marco principal para detectar rostros en imágenes.
    4. Tkinter: Librería de Python para crear interfaces gráficas.
    5. Pillow (PIL): Librería de Python para manipular imágenes.
    6. Pyttsx3: Librería de Python para realizar la vocalización de mensajes.
    7. theading: Librería de Python para crear hilos.
    8. Git: Sistema de control de versiones para el desarrollo del proyecto.
    9. Github: Sitio web para alojar el repositorio del proyecto.
    10. Google Colab: Plataforma para ejecutar código Python en la nube.
    11. Git LFS: Sistema de almacenamiento de archivos para el repositorio del proyecto.
    12. Visual Studio Code: Editor de código para el desarrollo del proyecto.
    13. Kaggle: Sitio web para descargar conjuntos de datos de aprendizaje automático.

4.  Resultados del Proyecto

    1.  El resultado del proyecto es una aplicación de escritorio funcional (main.py) llamada "VisualHelp", diseñada para ofrecer asistencia inmediata y audible sobre las personas presentes en una imagen. La aplicación utiliza la visión por computadora para identificar rostros familiares y desconocidos, y la vocalización de mensajes para comunicar esta información.
    2. Entrada: El usuario proporciona una imagen (vía URL o ruta local).
    3. Detección: La librería MTCNN localiza todos los rostros humanos en la imagen.
    4. Clasificación (CNN): Cada rostro detectado es alimentado a la CNN entrenada. El modelo clasifica el rostro como un "Familiar" específico (si la confianza supera el umbral (95%), o como "No Familiar".
    5. Salida Visual: La interfaz muestra la imagen con cuadros delimitadores de color: Verde para conocidos y Rojo para no familiares. El resultado se resume debajo con el formato "Rostro 1: PREDICCIÓN".
    6. Salida Audible (Accesibilidad): De forma automática, el sistema vocaliza el resultado completo (ej., "Análisis completado. Se detectaron dos rostros. Rostro número uno identificado como Juan Pérez, persona conocida..."). Esta predicción puede repetirse en cualquier momento con el botón "Escuchar Predicción".

INFORMACIÓN GENERAL DEL PROYECTO DE HACKATHON:
1. Planteamiento del Problema
    - Actualmente, el 71% las personas con discapacidad visual en el mundo, enfrentan barreras para acceder al mercado laboral debido a la dificultad de interactuar con el entorno físico en el puesto de trabajo.
    - El problema central no es la falta de capacidad del individuo, sino que el 80% de la información en un puesto de trabajo (señalética, ubicación de objetos, movimiento de personas) es puramente visual, lo que excluye automáticamente a quien no percibe estos datos.
    - Esta exclusión genera un ciclo de dependencia económica de terceros o del Estado.
    - Aunque existen herramientas como el bastón o lectores de pantalla para computadoras, falta una solución en tiempo real para el entorno físico dinámico, donde los objetos cambian de lugar constantemente.

2. Objetivos del Proyecto

    1. Dotar de autonomía al trabajador para que pueda moverse y realizar tareas en un entorno laboral
    2. Garantizar al usuario seguridad, mediante la detección de obstáculos y peligros
    3. Aumentar la posibilidades de trabajo para las personas con discapacidad visual
    4. Disminuir la dependencia social y económica de las personas con discapacidad visual
    5. Desarrollar una herramienta de soporte visual y auditivo que utilice la visión por computadora para interpretar el entorno y comunique esta información de manera audible, eliminando la barrera de la visión.

3. Herramientas utilizadas en el proyecto

    1. Python
    2. Git: Sistema de control de versiones para el desarrollo del proyecto.
    3. Github: Sitio web para alojar el repositorio del proyecto.
    4. Visual Studio Code: Editor de código para el desarrollo del proyecto.
    5. Flet: Framework de desarrollo de interfaces Web, Escritorio y Mobile utilizando exclusivemente Python.
    6. YOLO: Sistema de detección de objetos en imágenes

4.  Resultados del Proyecto
    1.  App Multiplataforma: Creación de una aplicación funcional para Web y Móvil utilizando Python y Flet.
    2. Precisión alta en la detección de objetos: Utilización de la librería YOLO para detectar objetos en imágenes.
    3. Audio  en directo: Utilización de la librería pyttsx3 para realizar la vocalización de mensajes.
    4. Asistencia en tiempo real: Utilización de modelo para realizar la identificación de voz y la interpretación de mensajes.