import matplotlib.pyplot as plt
import numpy as np
from numpy.random import uniform, randn
import nengo




def parametrize_learning_law(v0, vf, t0, tf, learning_type):
    assert learning_type in ('sigmoid', 'invtime', 'exp')

    #Matrix zur Speicherung der parametrisierten Lernrate für jeden Schritt
    y = np.zeros((tf - t0,))

    #Zeitschritte von 1 bis tf
    t = np.array([i for i in range(1, tf +1 )])

    #Unterscheidung nach Lernarten (sigmoid, invtime, exp)
    if learning_type == 'sigmoid':
        s = -np.floor(np.log10(tf)) * 10**(-(np.floor(np.log10(tf))))
        p = abs(s*10**(np.floor(np.log10(tf)) + np.floor(np.log10(tf))/2))
        y = v0 - (v0)/(1+np.exp(s*(t-(tf/p)))) + vf
    
    elif learning_type == 'invtime':
        B = (vf * tf - v0 * t0) / (v0 - vf)
        A = v0 * t0 + B * v0
        y = [A / (t[i] + B) for i in range(len(t))]
    
    elif learning_type == 'exp':
        if v0 < 1:
            p = -np.log(v0)
        else:
            p = np.log(v0)
        y = v0 * np.exp(-t/(tf/p))

    return y





#Anzahl Ensembles
N_ENS = 2

#Anzahl der Neuronen in SOM
N_NEURONS = 50

#Wertebereich
RADIUS = 1.

#Schrittweite der Nengo-Simulation
DT = .001

#Max Anzahl Epochen für inneres Lernen der SOM
MAX_EPOCHS_IN_LEARNING = 500

#Max Anzahl Epochen für äußeres Lernen der SOM
MAX_EPOCHS_XMOD_LEARNING = 1000

#Anzahl Samples für Training
N_SAMPLES = 1500

#Aktivitätsverfallrate
ETA = 1.

#Gewichtsverfallrate
XI = 1e-3

#Konstante, wie schnell Membranspannung bei fehlendem Eingang auf Null abfällt
TAU_RC = .02
#wie lange bleibt Membranspannung nach einem Spike auf Null
TAU_REF = .002

#Lernarten
LEARNING_TYPES = ['sigmoid', 'invtime', 'exp']

#Parametrisierung der Lernraten
sigma0 = N_NEURONS / 2. #ursprünglicher Wert
sigmaf = 1.             #finaler Wert
SIGMAT = parametrize_learning_law(v0=sigma0, vf=sigmaf, t0=1, tf=MAX_EPOCHS_IN_LEARNING, learning_type='exp')
alpha0 = .1             #ursprünglicher Wert
alphaf = .01           #finaler Wert
ALPHAT = parametrize_learning_law(v0=alpha0, vf=alphaf, t0=1, tf=MAX_EPOCHS_IN_LEARNING, learning_type='exp')

#Vorhandene Sensoren
SENSORES = ['x', 'y']
assert len(SENSORES) == N_ENS

ACTIVITY = {}
for i in range(N_ENS):
    ACTIVITY[i] = np.zeros((N_NEURONS))

# Matrizen für Inputgewichte zwischen den Neuronen werden abgespeichert, um als transform-Parameter auf Ensembles angewandt werden zu können
INPUT_WEIGHTS = {}
for i, _ in enumerate(SENSORES):
    INPUT_WEIGHTS[i] = np.zeros((N_NEURONS,))

# Standartabweichungen zur Berechnungen der Gewichtsänderungen
SIGMA_DEF =.045
STD = {}
for i, _ in enumerate(SENSORES):
    STD[i] = SIGMA_DEF * np.ones((N_NEURONS,))

# Matrizen für X-Korrelation zwischen den Neuronen werden abgespeichert, um als transform-Parameter auf Ensembles angewandt werden zu können
W_CROSS = uniform(0, 1, (N_NEURONS, N_NEURONS))
XMOD_WEIGHTS = {}
for i, _ in enumerate(SENSORES):
    XMOD_WEIGHTS[i] = W_CROSS / W_CROSS.sum()



#setzt Parameter zurück zur Wiederverwendung
def reset_matrices():
    for i, _ in enumerate(SENSORES):
        ACTIVITY[i] = np.zeros((N_NEURONS,))
        INPUT_WEIGHTS[i] = np.zeros((N_NEURONS,))
        STD[i] = SIGMA_DEF * np.ones((N_NEURONS,))
        XMOD_WEIGHTS[i] = W_CROSS / W_CROSS.sum()



def generate_test_data():
    data = {}

    data['x'] = np.array(np.random.uniform(-1,1, N_SAMPLES))
    data['y'] = np.array([x**3 for _,x in enumerate(data['x'])])

    return data


data = generate_test_data()
SENSORY_DATA = np.column_stack((data['x'], data['y']))



class LearnProcessBase(nengo.Process):
    def make_step(self, shape_in, shape_out, dt, rng, state=None):
        def step(t, x):
            epoch = int(t/DT)

            if epoch == 0: #wenn erster Durchlauf, Werte der Neuronen initialisieren
                winputs = np.zeros((N_ENS, N_NEURONS))
                activities = np.zeros((N_ENS, N_NEURONS))
                stds = np.ones((N_ENS, N_NEURONS)) * SIGMA_DEF
                #xmod_weights = W_CROSS / W_CROSS.sum()
            else:
                winputs = np.array([x[:N_NEURONS], x[N_NEURONS:N_NEURONS*2]])
                activities = np.array([x[N_NEURONS*2:N_NEURONS*3], x[N_NEURONS*3:N_NEURONS*4]])
                stds = np.array([x[N_NEURONS*4:N_NEURONS*5], x[N_NEURONS*5:N_NEURONS*6]])
                #xmod_weights = np.array(x[N_NEURONS*6:]).reshape((N_NEURONS, N_NEURONS))
            
            #Inneres Lernen
            if epoch <= MAX_EPOCHS_IN_LEARNING:
                (winputs, activities, stds) = self.inner_learning(winputs, activities, stds, epoch)

            #XMOD-Lernen
            activities = self.xmod_learning(winputs, activities, stds, epoch)

            return np.concatenate((winputs.flatten(), activities.flatten(), stds.flatten()), axis=None)
        return step
    
    #für Ansatz, welcher winput, activity und std je als Ensemble implementiert
    def update_activity_vector(self, winput, activity, std, datapoint):
        #Initialisierung des Aktivitätsvektors auf 0
        act_cur = np.zeros((N_NEURONS,))

        #hier kann teilen durch 0 auftreten, weshalb 0-Werte ersetzt werden
        std = np.maximum(std, 1e-4)
        act_cur = (1 / (np.sqrt(2 * np.pi) * std)) * np.exp(-np.square(datapoint - winput) / (2 * np.square(std)))

        #Normalisierung des Aktivitätsvektors der Population
        if act_cur.sum() != 0:
            act_cur /= act_cur.sum()

        #Aktualisierung der Aktivität für die nächste Iteration
        activity = (1 - ETA) * activity + ETA * act_cur
        return activity
    
    #für Ansatz, welcher winput, activity und std je als Ensemble implementiert
    def inner_learning(self, winputs, activities, stds, epoch):
        #Kernelwert (Differenz zw. Pos. des aktuellen Neurons und der Pos. des Neurons mit der höchsten Aktivität)
        hwi = np.zeros((N_NEURONS,))

        #Werte der Lernraten bleiben bei Schleifendurchläufe konstant und müssen so nur einmal gelesen werden
        alpha = ALPHAT[epoch-1]
        sigma = SIGMAT[epoch-1]

        #Faktor ändert sich nicht in Schleifendurchläufen und muss nur einmal berechnet werden
        factor = 1 / (np.sqrt(2 * np.pi) * sigma)

        for _, datapoint in enumerate(SENSORY_DATA): #iterieren über Datenpunkte
            for i in range(len(SENSORES)):
                # update the activity for the next iteration
                activities[i] = self.update_activity_vector(winputs[i], activities[i], stds[i], datapoint[i])

                #Bestimmung des Neurons mit höchster Aktivität (Gewinnerneuron)
                win_pos = np.argmax(activities[i])

                #Berechnung des Kernelwerts für Neuronen
                #einfacher Gauß'scher Kernel ohne Berücksichtigung der Begrenzungen
                hwi = np.exp(-np.square(np.arange(N_NEURONS) - win_pos) / (2 * sigma**2))

                #Aktualisierung der Gewichtungen der Neuronen
                winputs[i] += alpha * hwi * (datapoint[i] - winputs[i])

                #Aktualisierungen der std der Neuronen
                stds[i] += alpha * factor * hwi * ((datapoint[i] - winputs[i])**2 - stds[i]**2)
        return (winputs, activities, stds)
    
    def xmod_learning(self, winputs, activities, stds, epoch):
        raise NotImplementedError
    

class CovarianceLearning(LearnProcessBase):    
    #Covariance-Learning für Ansatz, welcher winput, activity und std je als Ensemble implementiert
    def xmod_learning(self, winputs, activities, stds, epoch):
        #  mean activities for covariance learning
        avg_act = np.zeros((N_NEURONS, N_ENS))

        # Berechnung des Abfalls für den Mittelwert
        omega = .002 + .998 / (epoch + 2)

        for _, datapoint in enumerate(SENSORY_DATA):
            for i in range(N_ENS):
                #Aktualisierung des Aktivitätsvektor
                activities[i] = self.update_activity_vector(winputs[i], activities[i], stds[i], datapoint[i])
                # Berechnung des Abfalls für den Mittelwert
                avg_act[:, i] = (1 - omega) * avg_act[:, i] + omega * activities[i][:]

            #Kreuzmodale Hebb'sche Kovarianz-Lernregel: Aktualisierung der Gewichte basierenf auf Kovarianz
            XMOD_WEIGHTS[0] = (1 - XI) * XMOD_WEIGHTS[0] + XI * (activities[0] - \
                              avg_act[:, 0].reshape(N_NEURONS,1)) * (activities[1] - avg_act[:,1].reshape(N_NEURONS,1)).T
            #xmod_weights = (1 - XI) * xmod_weights + XI * (activities[0] - \
            #               avg_act[:, 0].reshape(N_NEURONS,1)) * (activities[1] - avg_act[:,1].reshape(N_NEURONS,1)).T
            XMOD_WEIGHTS[1] = (1 - XI) * XMOD_WEIGHTS[1] + XI * (activities[1] - \
                              avg_act[:, 1].reshape(N_NEURONS,1)) * (activities[0] - avg_act[:, 0].reshape(N_NEURONS,1)).T
        return activities


class HebbianLearning(LearnProcessBase):
    #Hebbian-Learning für Ansatz, welcher winput, activity und std je als Ensemble implementiert
    def xmod_learning(self, winputs, activities, stds, epoch):
        for _, datapoint in enumerate(SENSORY_DATA):
            #Aktualisierung des Aktivitätsvektor
            for i in range(N_ENS):
                activities[i] = self.update_activity_vector(winputs[i], activities[i], stds[i], datapoint[i])

            #Hebb'sche Regel für Kreuzmodalität: Multiplikation der Aktivitäten
            XMOD_WEIGHTS[0] = (1 - XI) * XMOD_WEIGHTS[0] + XI * activities[0] * activities[1].T
            #xmod_weights = (1 - XI) * xmod_weights + XI * activities[0] * activities[1].T
            XMOD_WEIGHTS[1] = (1 - XI) * XMOD_WEIGHTS[1] + XI * activities[1] * activities[0].T
        return activities
    

class OjaLearning(LearnProcessBase):
    #Oja-Learning für Ansatz, welcher winput, activity und std je als Ensemble implementiert
    def xmod_learning(self, winputs, activities, stds, epoch):
        for _, datapoint in enumerate(SENSORY_DATA):
            #Aktualisierung des Aktivitätsvektor
            for i in range(N_ENS):
                activities[i] = self.update_activity_vector(winputs[i], activities[i], stds[i], datapoint[i])

            # Oja'sche lokale PCA-Lernregel
            XMOD_WEIGHTS[0] = ((1 - XI) * XMOD_WEIGHTS[0] + XI * activities[0] * activities[1].T) / \
                              np.sqrt(sum(sum((1 - XI) * XMOD_WEIGHTS[0] + XI * activities[0] * activities[1].T)))
            #xmod_weights = ((1 - XI) * xmod_weights + XI * activities[0] * activities[1].T) / \
            #               np.sqrt(sum(sum((1 - XI) * xmod_weights + XI * activities[0] * activities[1].T)))
            XMOD_WEIGHTS[1] = ((1 - XI) * XMOD_WEIGHTS[1] + XI * activities[1] * activities[0].T) / \
                              np.sqrt(sum(sum((1 - XI) * XMOD_WEIGHTS[1] + XI * activities[1] * activities[0].T)))
        return activities



if __name__ == "__main__":
    reset_matrices()

    with nengo.Network() as net:
        #Repräsentation der WInput-Vektoren als Ensembles
        winput_x = nengo.networks.EnsembleArray(N_NEURONS*60, n_ensembles=N_NEURONS, radius=RADIUS, neuron_type=nengo.LIF(tau_rc=TAU_RC, tau_ref=TAU_REF))
        winput_y = nengo.networks.EnsembleArray(N_NEURONS*60, n_ensembles=N_NEURONS, radius=RADIUS, neuron_type=nengo.LIF(tau_rc=TAU_RC, tau_ref=TAU_REF))

        #Repräsentation der Aktivitäts-Vektoren als Ensembles
        activity_x = nengo.networks.EnsembleArray(N_NEURONS*60, n_ensembles=N_NEURONS, radius=RADIUS, neuron_type=nengo.LIF(tau_rc=TAU_RC, tau_ref=TAU_REF))
        activity_y = nengo.networks.EnsembleArray(N_NEURONS*60, n_ensembles=N_NEURONS, radius=RADIUS, neuron_type=nengo.LIF(tau_rc=TAU_RC, tau_ref=TAU_REF))

        #Repräsentation der std-Vektoren als Ensembles
        std_x = nengo.networks.EnsembleArray(N_NEURONS*60, n_ensembles=N_NEURONS, radius=RADIUS, neuron_type=nengo.LIF(tau_rc=TAU_RC, tau_ref=TAU_REF))
        std_y = nengo.networks.EnsembleArray(N_NEURONS*60, n_ensembles=N_NEURONS, radius=RADIUS, neuron_type=nengo.LIF(tau_rc=TAU_RC, tau_ref=TAU_REF))

        #xmod_weights = nengo.networks.EnsembleArray(N_NEURONS*N_NEURONS*50, n_ensembles=N_NEURONS*N_NEURONS, radius=RADIUS)

        learn_node = nengo.Node(CovarianceLearning(), size_in=N_NEURONS*N_ENS*3, size_out=N_NEURONS*N_ENS*3)

        #Verbindungen von Neuronen der Ensembles zur Lernmethode ...
        nengo.Connection(winput_x.output, learn_node[:N_NEURONS])
        nengo.Connection(winput_y.output, learn_node[N_NEURONS:N_NEURONS*2])
        nengo.Connection(activity_x.output, learn_node[N_NEURONS*2:N_NEURONS*3])
        nengo.Connection(activity_y.output, learn_node[N_NEURONS*3:N_NEURONS*4])
        nengo.Connection(std_x.output, learn_node[N_NEURONS*4:N_NEURONS*5])
        nengo.Connection(std_y.output, learn_node[N_NEURONS*5:N_NEURONS*6])
        #nengo.Connection(xmod_weights.output, learn_node[N_NEURONS*6:])
        # ... und wieder zurück
        nengo.Connection(learn_node[:N_NEURONS], winput_x.input)
        nengo.Connection(learn_node[N_NEURONS:N_NEURONS*2], winput_y.input)
        nengo.Connection(learn_node[N_NEURONS*2:N_NEURONS*3], activity_x.input)
        nengo.Connection(learn_node[N_NEURONS*3:N_NEURONS*4], activity_y.input)
        nengo.Connection(learn_node[N_NEURONS*4:N_NEURONS*5], std_x.input)
        nengo.Connection(learn_node[N_NEURONS*5:N_NEURONS*6], std_y.input)
        #nengo.Connection(learn_node[N_NEURONS*6:], xmod_weights.input)


        #Proben
        p_winput_x = nengo.Probe(winput_x.output)
        p_winput_y = nengo.Probe(winput_y.output)
        p_activity_x = nengo.Probe(activity_x.output)
        p_activity_y = nengo.Probe(activity_y.output)
        p_std_x = nengo.Probe(std_x.output)
        p_std_y = nengo.Probe(std_y.output)
        #p_xmod_weights = nengo.Probe(xmod_weights.output)

    with nengo.Simulator(net, dt=DT) as s:
        s.run_steps(MAX_EPOCHS_XMOD_LEARNING)

    plt.figure(figsize=(12, 12))
    #plt.subplot(10, 1, 1)
    #plt.bar(range(N_NEURONS), s.data[p_activity_x][-1])
    #plt.title('activity_x')
    #plt.xlabel('Index')
    #plt.ylabel('Wert')

    #plt.subplot(10, 1, 2)
    #plt.bar(range(N_NEURONS), s.data[p_activity_y][-1])
    #plt.title('activity_y')
    #plt.xlabel('Index')
    #plt.ylabel('Wert')

    #plt.subplot(10, 1, 3)
    #plt.bar(range(N_NEURONS), s.data[p_winput_x][-1])
    #plt.title('winput_x')
    #plt.xlabel('Index')
    #plt.ylabel('Wert')

    #plt.subplot(10, 1, 4)
    #plt.bar(range(N_NEURONS), s.data[p_winput_y][-1])
    #plt.title('winput_y')
    #plt.xlabel('Index')
    #plt.ylabel('Wert')

    #plt.subplot(10, 1, 5)
    #plt.grid()
    #x = np.linspace(-RADIUS, RADIUS, N_NEURONS)
    #for i in range(N_NEURONS):
    #    # extract the preferred values (wight vector) of each neuron
    #    v_pref = s.data[p_winput_x][i]
    #    fx = np.exp(-(x - v_pref)**2 / (2 * s.data[p_std_x][i]**2))
    #    plt.plot([x for x in range(N_NEURONS)], fx)

    #plt.subplot(10, 1, 6)
    #plt.grid()
    #x = np.linspace(-RADIUS, RADIUS, N_NEURONS)
    #for i in range(N_NEURONS):
    #    # extract the preferred values (wight vector) of each neuron
    #    v_pref = s.data[p_winput_y][i]
    #    fx = np.exp(-(x - v_pref)**2 / (2 * s.data[p_std_y][i]**2))
    #    plt.plot([x for x in range(N_NEURONS)], fx)
    #
    #plt.subplot(10, 1, 7)
    #plt.bar(range(N_NEURONS), s.data[p_std_x][-1])
    #plt.title('std_x')
    #plt.xlabel('Index')
    #plt.ylabel('Wert')

    #plt.subplot(10, 1, 8)
    #plt.bar(range(N_NEURONS), s.data[p_std_y][-1])
    #plt.title('std_y')
    #plt.xlabel('Index')
    #plt.ylabel('Wert')

    plt.subplot(2, 1, 1)
    plt.imshow(XMOD_WEIGHTS[0], cmap='viridis')
    plt.colorbar() 
    plt.title('XMOD_WEIGHTS x') 
    plt.xlabel('Spaltenindex')
    plt.ylabel('Zeilenindex')

    plt.subplot(2, 1, 2)
    plt.imshow(XMOD_WEIGHTS[1], cmap='viridis')
    plt.colorbar() 
    plt.title('XMOD_WEIGHTS y') 
    plt.xlabel('Spaltenindex')
    plt.ylabel('Zeilenindex')

    plt.show()

