from colorama import Fore, Back, Style
from pathlib import Path
from datetime import datetime

def pTitle(cad): # Print normal message
	print(Fore.LIGHTYELLOW_EX + cad + Style.RESET_ALL)

def pTitle2(cad): # Print normal message
	print(Fore.LIGHTRED_EX + cad + Style.RESET_ALL)

def moveColInplace(df, col, pos):
    col = df.pop(col)
    df.insert(pos, col.name, col)

def creadir(ruta):
	# logdir = 'Logs/' + datetime.now().strftime("%Y%m%d")
	logdir = ruta + '/' + datetime.now().strftime("%Y%m%d")
	Path(logdir).mkdir(parents=True, exist_ok=True)
	pTitle2("Saving plots directory: " + logdir)
	return logdir