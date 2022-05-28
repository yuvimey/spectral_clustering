import webbrowser
import time
from invoke import task, call

MAX_ITER = 300

MAX_K_3D = 12
MAX_N_3D = 500
MAX_K_2D = 12
MAX_N_2D = 500

EPSIlION = 0.0001

@task(aliases=['del'])
def delete(c, extra=''):

	# Deletes previous capi module

    patterns = ['*capi*.so']
    if extra:
        patterns.append(extra)
    for pattern in patterns:
        c.run("rm -rf {}".format(pattern))

@task(delete)
def build(c):
	
	#	Setup the C-API module
	
    c.run("python3.8.5 setup.py build_ext --inplace")


@task(build)
def run(c, k=0, n=0, Random=True): # CHANGE D
    print("---maximum capacity for 3-dimentional data:---")
    print("         K=%i    n=%i"%(MAX_K_3D, MAX_N_3D))
    print("--------------------------------------------")
    print("---maximum capacity for 2-dimentional data:---")
    print("         K=%i    n=%i"%(MAX_K_2D, MAX_N_2D))
    print("--------------------------------------------")
    c.run("python3.8.5 main.py %i %i %s" % (k, n, str(Random)))