import gym
import tensorflow as tf
from tensorflow import keras
import random
import numpy as np
import datetime as dt
import math

STORE_PATH = '.\\'
BATCH_SIZE = 32

# Evolucion de epsilon - el factor aleatorio en la respuesta
MAX_EPSILON = 1
MIN_EPSILON = 0.01
EPSILON_MIN_ITER = 5000
#Cuantas muestras deben haberse procesado antes de empezar a entrenar las NN
DELAY_TRAINING = 300

#Velocidad con la que transferimos el conocimiento de NN primary a NN target
TAU = 0.08

#Tasa de descuento
GAMMA = 0.95

RANDOM_REWARD_STD = 1.0


env = gym.make("CartPole-v0")
state_size = 4
num_actions = env.action_space.n


#Modelo
class DQModel(keras.Model):
    #Dueling o Duble Q
    def __init__(self, hidden_size: int, num_actions: int, dueling: bool):
        super(DQModel, self).__init__()
        #Determinar si trabajeremos con una Double Q o Dueling Q 
        self.dueling = dueling

        #Dos capas comunes
        self.dense1 = keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.dense2 = keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        #Salida que determina por cada accion el valor de Q: A(s,a)
        self.adv_dense = keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
        self.adv_out = keras.layers.Dense(num_actions,
                                          kernel_initializer=keras.initializers.he_normal())
        #Si usamos la estrategia de Double Q, esto seria todo. Si usamos Dueling Q, añadimos la parte del modelo
        #que modelo Q: V(s)
        if dueling:
            #En caso de trabajar en modo Dueling, añadimos la modelizacion de V(s). Lo primero es una capa normal...
            self.v_dense = keras.layers.Dense(hidden_size, activation='relu',
                                         kernel_initializer=keras.initializers.he_normal())
            #Con otra capa con una salida que no depende de las acciones
            self.v_out = keras.layers.Dense(1, kernel_initializer=keras.initializers.he_normal())
            #Como se indica en el articulo, cuando trabajamos en modo dueling, la salida sera:
            #V(s) -> salida de la capa self.v_out
            #A(s,a) -> salida de la capa self.adv_out 
            #menos el valor media de A(s,a) para el conjunto de las acciones. 
            #En el caso del dueling, Q(s,a) sera self.v_out + self.lambda_layer(self.adv_out)
            #Esta capa no tiene pesos que aprender
            self.lambda_layer = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
            self.combine = keras.layers.Add()

    def call(self, input):
        #Calculamos A(s,a)
        x = self.dense1(input)
        x = self.dense2(x)
        adv = self.adv_dense(x)
        adv = self.adv_out(adv)
        
        #Si estamos en modo Double, A(s,a) es Q(s,a), no hay V(s)
        #Si estamos en modo dueling...
        if self.dueling:
            #Calculamos V(s)
            v = self.v_dense(x)
            v = self.v_out(v)
            #Calculamos A(s,a) normalizado. Esto es, A(s,a) menos la media de A(s,a) a lo largo de todas las acciones
            #A(s,a) normalizado es la salida de la capa self.lambda_layer. 
            norm_adv = self.lambda_layer(adv)
            #Q(s,a) sea A(s,a) normalizado mas V(s)
            combined = self.combine([v, norm_adv])
            #Retorna Q(s,a)
            return combined
        #Retorna Q(s,a)
        return adv

#Si pasamos False trabajeramos en modo Double. Si pasamos True, en modo dueling
primary_network = DQModel(30, num_actions, True)
target_network = DQModel(30, num_actions, True)

#En modo dueling y en modo double, solo se adiestra la NN primary
primary_network.compile(optimizer=keras.optimizers.Adam(), loss='mse')

#Al incicio de proceso hacemos que la NN primary y target sean iguales
# make target_network = primary_network
for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
    t.assign(e)

#Helper para copiar la NN primary en la NN target. Notese como se incluye una inercia en la transferencia. La inercia la fija TAU
def update_network(primary_network, target_network):
    # update target network parameters slowly from primary network
    for t, e in zip(target_network.trainable_variables, primary_network.trainable_variables):
        t.assign(t * (1 - TAU) + e * TAU)

#Este objeto contiene vectors de entrenamiento
#El objecto tiene un diccionario de tamaño max_memory
#Cada vector de entrenamiento es de rango 4
#cada una de las cuatro features es:
# - states
# - actions
# - rewards
# - next_states
class Memory:
    #Crea un diccionario de tamaño max_memory 
    def __init__(self, max_memory):
        self._max_memory = max_memory
        self._samples = []
        
    #Añade un vector al diccionario
    def add_sample(self, sample):
        self._samples.append(sample)
        if len(self._samples) > self._max_memory:
            self._samples.pop(0)
            
    #recupera un batch de vectores de tamaño no_samples. Esto es, retorna un (no_samples,4)  
    def sample(self, no_samples):
        if no_samples > len(self._samples):
            return random.sample(self._samples, len(self._samples))
        else:
            return random.sample(self._samples, no_samples)

    @property
    def num_samples(self):
        return len(self._samples)

#Crea el diccionario de vectores de entrenamiento
memory = Memory(500000)

#Helper para elegir una accion. La policy es elegir la accion que maximiza Q(s,a) para un determinado estado
#Soporta una eleccion aleatoria con probabilidad eps
#Usa la NN primaria como modelo que implementa Q(s,a)
def choose_action(state, primary_network, eps):
    if random.random() < eps:
        return random.randint(0, num_actions - 1)
    else:
        return np.argmax(primary_network(state.reshape(1, -1)))

#Helper de entrenamiento
def train(primary_network, memory, target_network):
    #Recupera un batch de vectores
    batch = memory.sample(BATCH_SIZE)
    
    #Crea cuatro diccionarios con las cuatro features del vector
    states = np.array([val[0] for val in batch])
    actions = np.array([val[1] for val in batch])
    rewards = np.array([val[2] for val in batch])
    next_states = np.array([(np.zeros(state_size)
                             if val[3] is None else val[3]) for val in batch])
    
    #Como la funcion que tenemos que aprender es Q(s,a), es decir, necesitamos pasar al training
    #1. El vector de estado. Eso es facil, es states
    #2. El Q que obtendriamos con cada accion si estamos en el estado states. Bien, para todas las acciones que no hemos
    #elegido en este punto tomamos la propia salida de NN primary - es decir, nada que aprender para estas acciones
    #pero para la accion elegida, tomaremos el valor de la NN primary en next_states mas el reward
    #Empecemos
    
    #Estimacion Q(states,a). Sera un (BATCH_SIZE, num_actions)
    prim_qt = primary_network(states)
    #Entonces el target sera prim_qtm excepto para la accion que hemos elegido en este estado
    target_q = prim_qt.numpy()
    
    #Veamos que vectores del batch nos serven para el aprendizaje. Nos sirven solamente aquellos que tengan un next_state valido 
    #states es un vector (num_actions) con hot-encoding. Si la suma es cero significa que no hay next state, o lo que es lo mismo
    #hemos llegado al final del entrenamiento
    # valid_idxs sera un (BATCH_SIZE) de booleanos, con True en aquellos casos en los que haya un next state
    valid_idxs = np.array(next_states).sum(axis=1) != 0
    
    # Obtenemos Q(next_State,action) usando NN target
    q_from_target = target_network(next_states)
    
    # Necesitamos saber cual es la accion que usariamos en next_state. Usaremos la NN primary en next_state para 
    #identificar cual es la accion que maximizaria el valor
    #Estimacion Q(next_states,a). Sera un (BATCH_SIZE, num_actions)
    prim_qtp1 = primary_network(next_states)
    # (BATCH_SIZE)
    #Contiene la accion que se eligiria en next_state
    prim_action_tp1 = np.argmax(prim_qtp1.numpy(), axis=1)
    
    #Updates sera igual a low rewards, excepto en 
    #a) aquellos vectores que sen validos, esto es, que tengan un next state
    #b) en esos casos r + GAMMA * Q(next_state,action) usando la NN target para hacer el calculo
    #       de todas las acciones posibles en next_state, usaremos la que nos de una valor mayor   
    #((BATCH_SIZE, 1)
    updates = rewards
    batch_idxs = np.arange(BATCH_SIZE)
    updates[valid_idxs] += GAMMA * q_from_target.numpy()[batch_idxs[valid_idxs], prim_action_tp1[valid_idxs]]
    
    # Actualizamos el target_q
    target_q[batch_idxs, actions] = updates
    
    #Entrena la NN primaria con la entrada states, y el Q(s,a) esperado target_q 
    loss = primary_network.train_on_batch(states, target_q)
    return loss


num_episodes = 1000000
eps = MAX_EPSILON
render = False
train_writer = tf.summary.create_file_writer(STORE_PATH + f"/DuelingQ_{dt.datetime.now().strftime('%d%m%Y%H%M')}")
steps = 0
for i in range(num_episodes):
    cnt = 1
    avg_loss = 0
    tot_reward = 0
    state = env.reset()
    while True:
        if render:
            env.render()

        #El agente elige una accion y la ejecuta
        action = choose_action(state, primary_network, eps)
        next_state, _, done, info = env.step(action)
        
        #En este entorno la reward es siempre 1 cuando seguimos manteniendo el palo dentro de los parametros objetivo
        reward = np.random.normal(1.0, RANDOM_REWARD_STD)
        tot_reward += reward
        if done:
            next_state = None
        
        # Guardamos en el diccionario este vector
        memory.add_sample((state, action, reward, next_state))

        #Si hemos ejecutado ya el numero de steps minimos, empezamos el entrenamiento de las NN
        if steps > DELAY_TRAINING:
            loss = train(primary_network, memory, target_network)
            update_network(primary_network, target_network)
        else:
            loss = -1
        avg_loss += loss

        # linearly decay the eps value
        if steps > DELAY_TRAINING:
            eps = MAX_EPSILON - ((steps - DELAY_TRAINING) / EPSILON_MIN_ITER) * \
                  (MAX_EPSILON - MIN_EPSILON) if steps < EPSILON_MIN_ITER else \
                MIN_EPSILON
        steps += 1

        if done:
            if steps > DELAY_TRAINING:
                avg_loss /= cnt
                print(f"Episode: {i}, Reward: {cnt}, avg loss: {avg_loss:.5f}, eps: {eps:.3f}")
                with train_writer.as_default():
                    tf.summary.scalar('reward', cnt, step=i)
                    tf.summary.scalar('avg loss', avg_loss, step=i)
            else:
                print(f"Pre-training...Episode: {i}")
            break

        state = next_state
        cnt += 1

