#!/usr/bin/python3
import sys
import os

#$a = docker images - q
#$b =$a.ForEach({docker inspect - -format='{{.Id}} {{.Parent}}' $_})

def procesaImagen(s):
    s = s.rstrip()
    return s.replace('sha256:', '')#.split(' ')

if __name__ == '__main__':
    image_id = sys.argv[1]
    imagenes = os.popen('docker images -q --all').readlines()
    diccionario={}

    for imagen in map(lambda s: s.replace('sha256:', '').rstrip(), imagenes):
        #dependencias = 'docker inspect --format=\"{{.Id}} {{.Parent}}\" %s' % (imagen)
        dependencias = 'docker inspect --format=\"{{.Parent}}\" %s' % (imagen)
       
        dep=list(map(procesaImagen,os.popen(dependencias).readlines()))[0][:12]
        if len(dep)==0:
            continue
        if dep in diccionario:
            diccionario[dep] = diccionario[dep]+[imagen]
        else:
            diccionario[dep] =[imagen]

    if not image_id in diccionario:
        print("La imagen no tiene hijos. La borramos directamente")
        imagenes = os.popen("docker rmi -f {}".format(image_id)).readlines()
        print(imagenes)
    else:
        print("La imagen tiene hijos. Borramos primero los hijos, y luego la propia imagen")
        termina = False

        respuesta = diccionario[image_id][0]
        imagenes_a_borrar = [image_id, respuesta]
        while not termina:
            if not respuesta in diccionario:
                termina = True
            else:
                respuesta = diccionario[respuesta][0]
                imagenes_a_borrar = imagenes_a_borrar +[respuesta]
            
        for a in reversed(imagenes_a_borrar):
            imagenes = os.popen("docker rmi -f {}".format(a)).readlines()
            print(imagenes)
    
