import pygame

pygame.init()
pygame.joystick.init()

print ("controllers: ", pygame.joystick.get_count())

joyStick = pygame.joystick.Joystick(0)
joyStick.init()

print ("name: ", joyStick.get_name())
print ("axes: ", joyStick.get_numaxes())

while True:
    pygame.event.pump()

    values = [
        joyStick.get_axis(i) for i in range(joyStick.get_numaxes())
    ]
    print("Axis values:", values, end="\r")