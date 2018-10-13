#lanciare da terminale
#ATTENZIONE condizione in line 162 settata a mano, dipende dal problema

from pylab import *
import numpy
from skimage import io
import sys
import pp

def chris(i,j,k,t,r,theta,phi,m):
	temp=0
	if j>k: #simmetric
		temp=j
		j=k
		k=temp
		
	if i==0:
		if j==0:
			if k==0:
				return (2*m**2)/r**3
			elif k==1:
				return (m*(2*m + r))/r**3
			else:
				return 0
		elif j==1:
			if k==1:
				return (2*m*(2*m + r))/r**3 - (2*m**2)/r**3
			else:
				return 0
		elif j==2:
			if k==2:
				return -2*m
			else:
				return 0
		else:
			return -2*m*numpy.sin(theta)**2

	elif i==1:
		if j==0:
			if k==0:
				return -(m*(2*m - r))/r**3
			elif k==1:
				return -(2*m**2)/r**3
			else:
				return 0
		elif j==1:
			if k==1:
				return (m*(2*m - r))/r**3 - (4*m**2)/r**3
			else:
				return 0
		elif j==2:
			if k==2:
				return 2*m - r
			else:
				return 0
		else:
			return numpy.sin(theta)**2*(2*m - r)

	elif i==2:
		if j==0:
			return 0
		elif j==1:
			if k==2:
				return 1.0/r
			else:
				return 0
		elif j==2:
			return 0
		else:
			return -numpy.cos(theta)*numpy.sin(theta)
		
	else:
		if j==0:
			return 0
		elif j==1:
			if k==3:
				return 1.0/r
			else:
				return 0
		elif j==2:
			if k==2:
				return 0
			else:
				return numpy.cos(theta)/numpy.sin(theta)
		else:
			return 0

def ctos(x,y,z,xp,yp,zp):
	r=numpy.sqrt(x**2+y**2+z**2)
	theta=numpy.arccos(z/r)
	phi=numpy.arctan2(y,x)
	rp=numpy.sin(theta)*numpy.cos(phi)*xp + numpy.sin(theta)*numpy.sin(phi)*yp + numpy.cos(theta)*zp
	phip=-numpy.sin(phi)/r/numpy.sin(theta)*xp + numpy.cos(phi)/r/numpy.sin(theta)*yp
	thetap=numpy.cos(theta)*numpy.cos(phi)/r*xp + numpy.sin(phi)*numpy.cos(theta)/r*yp - numpy.sin(theta)/r*zp
	return r,theta,phi,rp,thetap,phip

def stoc(r,theta,phi):
	x=r*numpy.sin(theta)*numpy.cos(phi)
	y=r*numpy.sin(theta)*numpy.sin(phi)
	z=r*numpy.cos(theta)
	return x,y,z

def geo_S(lam, x, m):
	x_new=numpy.zeros(len(x))
	x_new[0]=x[4]
	x_new[1]=x[5]
	x_new[2]=x[6]
	x_new[3]=x[7]
	
	for i in range (4,8):
		for j in range(4):
			for k in range(4):
				x_new[i]=x_new[i]-chris(i-4,j,k,x[0],x[1],x[2],x[3],m)*x[j+4]*x[k+4]
	return x_new

def runge_45(h, hlim, errlim, x, m): #h: incremento; hlim: array 1x2 con liminf e limsup; errlim: analogo array 2x2; incond: array 1x8 con spazio/tempo (4) e velocita'/tempo (4); geo_S: nome funzione che valuta
	lam=1
	enter=True
	while(enter):
		k1 = h*geo_S(lam, x, m)
		k2 = h*geo_S(lam + 0.25*h, x + 0.25*k1, m)
		k3 = h*geo_S(lam + 3./8.*h, x + 3./32.*k1 + 9./32.*k2, m)
		k4 = h*geo_S(lam + 12./13.*h, x + 1932./2197.*k1 - 7200./2197.*k2 + 7296./2197.*k3, m)
		k5 = h*geo_S(lam + h, x + 439./216.*k1 - 8.*k2 + 3680./513.*k3 - 845./4104.*k4, m)
		k6 = h*geo_S(lam + 0.5*h, x - 8./27.*k1 + 2.*k2 - 3544./2565.*k3 + 1859./4104.*k4 - 11./40*k5, m)
		x_4 = x + 25./216.*k1 + 1408./2565.*k3 + 2197./4104.*k4 - 1./5.*k5 #soluzione al quart'ordine
		x_5 = x + 16./135.*k1 + 6656./12825.*k3 + 28561./56430.*k4 - 9./50.*k5 + 2./55.*k6 #soluzione al quint'ordine
		err = numpy.linalg.norm(x_4-x_5, numpy.inf) #valutazione errore
		enter=False
		if (err<errlim[0]) and (2*h<=hlim[1]):
			h=2*h
			enter=True
		elif (err>errlim[1]) and (0.5*h>=hlim[0]):
			h=0.5*h
			enter=True

	return x_4, h

def geo(jobnum,pix,ncpus,x_in,y_in,z_in,largh_image,x_box,y_box,z_box,dy_image,dz_image,coord_obj,hlim,errlim,m,h,N,jesoo,background_im):
	for l in range(pix): #loop che spara raggi riga per riga
		y_ini=y_in+jobnum*largh_image/ncpus #riparte da y_in ogni giro
		for f in range(pix/ncpus): #loop che spara raggi colonna per colonna
			
			##condizioni iniziali cartesiane
			x_cart = numpy.array([0., x_in, y_ini, z_in])
			u_cart = (coord_obj-x_cart)/numpy.linalg.norm(coord_obj-x_cart)
			x_mu = numpy.append(x_cart, u_cart)
			x_mu[4]=1.
			
			##trasformazioni in sferiche
			x_sfer=numpy.zeros(len(x_mu))
			x_sfer[1],x_sfer[2],x_sfer[3],x_sfer[5],x_sfer[6],x_sfer[7]=ctos(x_mu[1],x_mu[2],x_mu[3],x_mu[5],x_mu[6],x_mu[7])
			x_sfer[0]=x_mu[0]+2*m*numpy.log(x_sfer[1]-2*m) #da t a T
			x_sfer[4]=x_mu[4]+2.*m/(x_sfer[1]-2*m)*x_sfer[5]
			for i in range(1,N):
				x_sfer, h = runge_45(h, hlim, errlim, x_sfer, m)
				
				if (x_sfer[1]<2*m):
					#~ if m==0:
					break
				
				if x_sfer[1]>400.0: #ATTENZIONE dipendente dalla dimensione minima del box. Solo per risparmiare tempo
					x, y, z=stoc(x_sfer[1], x_sfer[2], x_sfer[3])
		
					if (y>=y_box*0.5) or (y<=-y_box*0.5) or (z>=z_box*0.5) or (z<=-z_box*0.5):
						break
					
					if int(x)>=x_box:	
						jesoo[l,f,:] = background_im[int(z)+z_box/2, int(y)+y_box/2, :] #associo al pixel del film quello dell'immagine di background
						break
	
			y_ini=y_ini+dy_image
		z_in=z_in-dz_image
	return jesoo

##parallel
ppservers = ()
job_server = pp.Server(ppservers=ppservers)
ncpus=job_server.get_ncpus() #number of cpus, 8 here

##valori iniziali
m=15. #input("Che massa vuoi dare al buco nero?\n")
h=1.
N=1500 #max iterations
hlim=array([0.01, 5])
errlim=([1e-10, 1e-5])
pix=16 #input("Quanti pixel vuoi fare per questa immagine?\n") #minimo 8 come le cpu

largh_image=2.
alt_image=2.

#for large images
#x_in=-550. #distanza dal buco nero lungo x del film
#y_in=-0.5*largh_image
#z_in=0.5*alt_image
#dist_obj=1.6
#coord_obj=array([0., x_in + dist_obj, 0., 0.]) #coordinate cartesiane dell'obiettivo
#x_box=500 #distanza dal buco nero lungo x del background. Change also line 162

#for small images
x_in=-100. #distanza dal buco nero lungo x del film
y_in=-0.5*largh_image
z_in=0.5*alt_image
dist_obj=0.7
coord_obj=array([0., x_in + dist_obj, 0., 0.]) #coordinate cartesiane dell'obiettivo
x_box=80. #Change also line 162. Distanza dal buco nero lungo x del background.

#immagine .jpg se no dice errore (con png)
background_im = imread("art.jpg")
jesoo = zeros((pix,pix/ncpus,3), uint8)

z_box=shape(background_im)[0] #righe
y_box=shape(background_im)[1] #colonne

##geodetiche
dy_image=largh_image/pix
dz_image=alt_image/pix

#l'immagine viene divisa in ncpus colonne
inputs=tuple(range(ncpus))
jobs = [(input, job_server.submit(geo, (input,pix,ncpus,x_in,y_in,z_in,largh_image,x_box,y_box,z_box,dy_image,dz_image,coord_obj,hlim,errlim,m,h,N,jesoo,background_im), (ctos, stoc, chris, geo_S, runge_45), ("numpy", ))) for input in inputs]

foto = zeros((pix,pix,3), uint8)
for input, job in jobs:
	foto[0:pix, input*pix/ncpus:(input+1)*pix/ncpus,:] = job()

job_server.print_stats()

#title="sun"+"_"+str(pix)+"_"+str(m)+".png"
imshow(foto)
axis('off')
#savefig(title)
savefig('mac3.png')
show()
