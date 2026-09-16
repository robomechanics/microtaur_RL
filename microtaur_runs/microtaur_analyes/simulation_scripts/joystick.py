import math
import pygame

DEADZONE = 0.15

FORWARD_MIN = 0.10
FORWARD_MAX = 0.20
YAW_MAX = 0.25


def remove_deadzone(value: float) -> float:
    if abs(value) <= DEADZONE:
        return 0.0

    magnitude = (abs(value) - DEADZONE) / (1.0 - DEADZONE)
    return math.copysign(magnitude, value)


def joystick_to_command(stick_x: float, stick_y: float):
    forward_input = max(0.0, -stick_y)

    if forward_input <= DEADZONE:
        vx = 0.0
    else:
        normalized = (
            forward_input - DEADZONE
        ) / (1.0 - DEADZONE)

        vx = (
            FORWARD_MIN
            + normalized * (FORWARD_MAX - FORWARD_MIN)
        )

    yaw_input = remove_deadzone(stick_x)
    yaw = -yaw_input * YAW_MAX

    return vx, yaw


pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    raise RuntimeError("No controller detected.")

controller = pygame.joystick.Joystick(0)
controller.init()

print("Controller:", controller.get_name())
print("Axes:", controller.get_numaxes())

while True:
    pygame.event.pump()

    stick_x = controller.get_axis(0)
    stick_y = controller.get_axis(1)

    vx, yaw = joystick_to_command(
        stick_x,
        stick_y,
    )

    print(
        f"stick=({stick_x:+.2f}, {stick_y:+.2f})  "
        f"vx={vx:+.3f} m/s  "
        f"yaw={yaw:+.3f} rad/s",
        end="\r",
    )