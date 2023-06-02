from colorama import Fore, Style
from colorama import init as colorama_init

from faceshine import appFaceShine

colorama_init()


def main():
    print(Fore.BLUE)
    print("*********************************")
    print("* Face Shine Service is running *")
    print("*********************************")
    print(Style.RESET_ALL)

    appFaceShine.run(debug=False, port='5000')

    print(Fore.GREEN)
    print("******************************")
    print("*      Service finished      *")
    print("******************************")
    print(Style.RESET_ALL)


if __name__ == '__main__':
    main()
