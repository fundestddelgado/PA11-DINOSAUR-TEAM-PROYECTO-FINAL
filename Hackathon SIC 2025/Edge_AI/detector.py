import cv2
from ultralytics import YOLO

from Audio.audio_engine import AudioEngine
from Audio.guide_logic import build_guidance


# =====================
# CONFIGURACIÓN
# =====================
MODEL_PATH = "yolov8n.pt"
CONF_THRESHOLD = 0.5


def main():
    print("Visual Support Work — Iniciando asistente")

    # =====================
    # CÁMARA
    # =====================
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ No se pudo abrir la cámara")
        return

    # =====================
    # AUDIO + MODELO
    # =====================
    audio = AudioEngine(cooldown=3)
    model = YOLO(MODEL_PATH)

    print("✅ Cámara y modelo listos")

    paused = False
    camera_ready = False

    # =====================
    # LOOP PRINCIPAL
    # =====================
    while True:

        ret, frame = cap.read()
        if not ret:
            print("❌ Falló lectura de cámara")
            break

        # 🔒 Asegurar que la cámara se vea antes de hablar
        if not camera_ready:
            camera_ready = True
            cv2.imshow("VSW Assistant", frame)
            cv2.waitKey(1)
            continue

        # =====================
        # TECLAS
        # =====================
        key = cv2.waitKey(30) & 0xFF

        if key == ord('p'):
            paused = not paused
            print("PAUSADO" if paused else "REANUDADO")

        elif key == ord('q'):
            break

        if paused:
            frame_pause = frame.copy()
            cv2.putText(
                frame_pause,
                "PAUSADO",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3
            )
            cv2.imshow("VSW Assistant", frame_pause)
            continue

        # =====================
        # DETECCIÓN
        # =====================
        height, width, _ = frame.shape
        results = model(frame, conf=CONF_THRESHOLD)

        guidance_spoken = False  # 🔇 hablar solo una vez por frame

        # =====================
        # 1️⃣ PRIORIDAD: PERSONA
        # =====================
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                label = model.names[int(box.cls[0])]

                if label != "person":
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_center = (x1 + x2) / 2
                box_height_ratio = (y2 - y1) / height

                guidance = build_guidance(
                    label=label,
                    x_center=x_center,
                    box_ratio=box_height_ratio,
                    frame_width=width
                )

                if guidance:
                    audio.speak(guidance)
                    guidance_spoken = True

                # 🎨 Visual
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                            (0, 255, 0), 2)
                break  # solo una persona

            if guidance_spoken:
                break

        # =====================
        # 2️⃣ SI NO HAY PERSONA → OBJETOS
        # =====================
        if not guidance_spoken:
            for r in results:
                if r.boxes is None:
                    continue

                for box in r.boxes:
                    label = model.names[int(box.cls[0])]

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    x_center = (x1 + x2) / 2
                    box_height_ratio = (y2 - y1) / height

                    guidance = build_guidance(
                        label=label,
                        x_center=x_center,
                        box_ratio=box_height_ratio,
                        frame_width=width
                    )

                    if guidance:
                        audio.speak(guidance)
                        guidance_spoken = True

                    # 🎨 Visual
                    cv2.rectangle(frame, (x1, y1), (x2, y2),
                                  (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (0, 255, 0), 2)
                    break

                if guidance_spoken:
                    break

        # =====================
        # MOSTRAR
        # =====================
        cv2.imshow("VSW Assistant", frame)

    # =====================
    # LIMPIEZA
    # =====================
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
