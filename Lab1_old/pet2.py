from math import *
import matplotlib.pyplot as plt

global E, R, Is, phi_t, eps, c, CONST_ALPHA
E = 5
R = 1
Is = 1.2E-10
phi_t = 0.025
eps = 0.001
c = 0.1
CONST_ALPHA = 0.1


def get_D(phi_i : float) -> float:
	numerator = (E - phi_i)/R - Is * (exp(phi_i/phi_t) - 1)
	denominator = - 1/R - Is/phi_t * exp(phi_i/phi_t)
	return -numerator/denominator
	
	
def get_alpha(D : float) -> float:
	if (abs(D) < c):
		return 1
	else:
		return c/abs(D)
		

def calc_classic(phi_0) -> (int, float):
	phi_next = 0
	phi_cur = phi_0
	i = 0
	while (True):
		i += 1
		D = get_D(phi_cur)
		phi_next = phi_cur + D
		print(i, '{:.2e}'.format(phi_next), '{:.2e}'.format(abs(D)), sep='\t')
		
		if (abs(D) < eps):	break
		phi_cur = phi_next
		
	return i, phi_next
		
		
def calc_accel(phi_0) -> (int, float):
	# sweep E from 0V to 5V
	global E
	phi_next = 0
	phi_cur = phi_0
	i = 0
	j = 0
	for E_loc in range(0,5+1, 1):
		E = E_loc
		(j, phi_next) = calc_classic(phi_cur)
		i += j
		print(E+1, '{:.2e}'.format(phi_next), abs(phi_next - phi_cur))
		phi_cur = phi_next
		
	return i, phi_next
		

def calc_w_const_alpha(phi_0) -> (int, float):
	phi_next = 0
	phi_cur = phi_0
	i = 0
	while (True):
		i += 1
		D = get_D(phi_cur)
		phi_next = phi_cur + CONST_ALPHA*D
		print(i, '{:.2e}'.format(phi_next), '{:.2e}'.format(abs(D)), sep='\t')

		if (abs(D) < eps):	break
		phi_cur = phi_next
		
	return i, phi_next
	
	
def calc_w_variable_alpha(phi_0) -> (int, float):
	phi_next = 0
	phi_cur = phi_0
	i = 0
	while (True):
		i += 1
		D = get_D(phi_cur)
		phi_next = phi_cur + get_alpha(D)*D
		print(i, '{:.2e}'.format(phi_next), '{:.2e}'.format(abs(D)), sep='\t')

		if (abs(D) < eps):	break
		phi_cur = phi_next

	return i, phi_next
	
	
############################################################################

def run_methods_for(phi_0):
	print('-'*5, '{:.2e}'.format(phi_0), 'classic', '-'*5)
	classic = calc_classic(phi_0)
	print("result =", classic)
	print(15*'-', end='\n\n')

	print('-'*5,'{:.2e}'.format(phi_0), 'acceler', '-'*5)
	accel = calc_accel(phi_0)
	print("result =", accel)
	print(15*'-', end='\n\n')

	print('-'*5,'{:.2e}'.format(phi_0), 'const a', '-'*5)
	w_const = calc_w_const_alpha(phi_0)
	print("result =", w_const)
	print(15*'-', end='\n\n')

	print('-'*7,'{:.2e}'.format(phi_0), 'var a', '-'*5)
	var = calc_w_variable_alpha(phi_0)
	print("result =", var)
	print(15*'-', end='\n\n')
	
	return [classic, accel, w_const, var]


if __name__ == '__main__':
	print("phi_0 = 0V")
	phi_0 = 0
	r1 = run_methods_for(phi_0)
	print()


	phi_0 = 0.6
	r2 = run_methods_for(phi_0)
	print()


	phi_0 = 0.7
	r3 = run_methods_for(phi_0)
	print()


	print("results table")
	print("phi_0", "method", "(i, res)", sep='\t')
	print(0, "classic", r1[0], sep='\t')
	print(0, "acceler", r1[1], sep='\t')
	print(0, "w const", r1[2], sep='\t')
	print(0, "variabl", r1[3], sep='\t')
	print('-'*20)

	print(0.6, "classic", r2[0], sep='\t')
	print(0.6, "acceler", r2[1], sep='\t')
	print(0.6, "w const", r2[2], sep='\t')
	print(0.6, "variabl", r2[3], sep='\t')
	print('-'*20)

	print(0.7, "classic", r3[0], sep='\t')
	print(0.7, "acceler", r3[1], sep='\t')
	print(0.7, "w const", r3[2], sep='\t')
	print(0.7, "variabl", r3[3], sep='\t')
