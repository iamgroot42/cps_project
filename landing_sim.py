from openap.traj import Generator
import matplotlib.pyplot as plt

def descent_trajectory(
        model_num, # Aircraft mdodel
        dt=1  # Timestep (seconds)
    ):

    trajgen = Generator(ac=model_num)
    
    # enable Gaussian noise in trajectory data
    trajgen.enable_noise()

    data_de = trajgen.descent(dt=dt, random=True)
    
    return data_de


def identify_phase_changes(segments):
    phases = {}
    for i, x in enumerate(segments):
        if x is None: continue
        if x not in phases:
            phases[x] = i

    # Dictionaries preserve order
    # Construct ranges in in [,) form
    keys = list(phases.keys())
    for i in range(len(keys)-1):
        phases[keys[i]] = (phases[keys[i]], phases[keys[i+1]])
    
    phases[keys[-1]] = (phases[keys[-1]], len(segments))
    return phases


if __name__ == "__main__":

    plane = 'b789'  # Boeing 787-9

    # descent  = descent_trajectory('a388') # Airbus A380-800
    # descent  = descent_trajectory('a320') # Airbus A-320
    # descent  = descent_trajectory('b77w') # Boeing 777-300ER

    n_tries = 3

    for _ in range(n_tries):
        descent  = descent_trajectory(plane)

        t, h, s, seg, v, vs = descent['t'], descent['h'], descent['s'], descent['seg'], descent['v'], descent['vs']
        phases = identify_phase_changes(seg)

        # Decision height cutoff: ~60 metres

        TIME, HEIGHT, = [], []
        # Our focus is on final approach
        for i in range(*phases['FA']):
            print(t[i], h[i], s[i], v[i], vs[i])
            TIME.append(t[i] - phases['FA'][0])
            HEIGHT.append(h[i])
    
        plt.plot(TIME, HEIGHT)
    plt.show()
    