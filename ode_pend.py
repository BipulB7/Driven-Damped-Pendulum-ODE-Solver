import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import seaborn as sns


class Pendulum:

    def __init__(self, Q=2.0, f0=1.6, Omega=2.0/3.0, tmax=6.0,
                 phi0=0.0, vphi0=0.0):
        self.Q = Q            # quality factor (inverse damping)
        self.f0 = f0          # forcing amplitude / mg
        self.Omega = Omega    # drive frequency / w0
        self.period = 2.0 * np.pi / self.Omega  # drive period
        # Initial values
        self.phi0 = phi0      # initial angular displacement
        self.vphi0 = vphi0    # initial angular velocity
        # Set start and stop times
        self.tmax = tmax      # maximum time in drive periods
        self.t_start = 0.0
        self.t_stop = tmax * 2.0 * np.pi / Omega  # dimensionless stopping time

    def solve(self, method="Radau", pts_per_period=200):
        def jac(t, y, Q, f0, Omega):  # Calculates Jacobian matrix
            phi, vphi = y             # unpack values of dep variables
            j11, j12 = 0.0, 1.0
            j21, j22 = -np.cos(phi), -1.0 / Q
            return np.array([[j11, j12], [j21, j22]])

        def f(t, y, Q, f0, Omega):
            phi, vphi = y      # unpack values of dep variables
            d_phi_dt = vphi    # calculate derivatives
            d_vphi_dt = -vphi / Q - np.sin(phi) + f0 * np.cos(Omega * t)
            return d_phi_dt, d_vphi_dt
        self.method = method
        psoln = solve_ivp(f,
                          [self.t_start, self.t_stop],
                          [self.phi0, self.vphi0],
                          args=(self.Q, self.f0, self.Omega),
                          method=self.method, jac=jac,
                          rtol=1e-6, atol=1e-8, dense_output=True)
        self.pts_per_period = pts_per_period
        t_increment = self.period / self.pts_per_period
        self.t = np.arange(self.t_start, self.t_stop, t_increment)
        self.z = psoln.sol(self.t)

    def plot(self, phidot=False, tmin=0, periodic=False, poincare=False):
        fig = plt.figure(figsize=(9.5, 7.5))
        c = sns.color_palette("icefire_r", 3)  # a Seaborn palette
        # Plot phi as a function of time
        ax1 = fig.add_subplot(211)
        if phidot is False:  # Plot phi vs time
            ax1.plot(self.t / self.period, self.z[0, :], color=c[0])
            ax1.set_ylabel(r"$\phi$", fontsize=14)
        else:  # Plot phidot vs time (good for rolling motion)
            ax1.plot(self.t / self.period, self.z[1, :], color=c[0])
            ax1.set_ylabel(r"$\dot\phi$", fontsize=14)
        ax1.set_xlabel(r"$t/T_\mathrm{drive}$", fontsize=14)
        ax1.axhline(lw=0.5, color="gray", zorder=-1)
        ax1.axvline(lw=0.5, color="gray", zorder=-1)

        # Make vphi vs phi phase-space plot (starting after tmin periods)
        istart = int(tmin) * self.pts_per_period
        ax2 = fig.add_subplot(223)

        if periodic or poincare:  # map phi onto -pi < phi < pi
            phi_periodic = (self.z[0, istart:] + np.pi) % (2.0 * np.pi) - np.pi
            phidot_periodic = self.z[1, istart:-1]
            if poincare:  # plot Poincare section on -pi < phi < pi
                ax2.plot(phi_periodic[10::self.pts_per_period],
                         phidot_periodic[10::self.pts_per_period],
                         ".", ms=1, color=c[2])
                txt = f"Poincaré section: {phi_periodic[10::200].size:d} points"
                ax2.set_title(txt)
            else:         # plot full phase-space trajectory on -pi < phi < pi
                diffphi = np.abs(phi_periodic[1:] - phi_periodic[:-1])
                # mask lines going from -pi to pi or pi to -pi
                phi_periodic = np.ma.masked_where(
                    diffphi > 3.0, phi_periodic[1:])
                ax2.plot(phi_periodic, phidot_periodic, "-", lw=0.75,
                         color=c[2])
        else:  # make normal phase-space plot
            ax2.plot((self.z[0, istart:]), self.z[1, istart:],
                     "-", lw=0.75, color=c[2])

        ax2.set_xlabel(r"$\phi$", fontsize=14)
        ax2.set_ylabel(r"$\dot\phi$", fontsize=14)
        ax2.axhline(lw=0.5, color="gray", zorder=-1)
        left, right = ax2.get_xlim()
        if (left <= 0.0) and (right >= 0.0):
            ax2.axvline(lw=0.5, color="gray", zorder=-1)
        # Write out parameters
        ax3 = fig.add_subplot(224)
        ax3.axis("off")
        txt = r"$Q = {0:0.1f}$".format(self.Q)
        txt += "\n" + r"$f_0 = {0:0.4f}$".format(self.f0)
        txt += "\n" + r"$\Omega = {0:0.4f}$".format(self.Omega)
        ax3.text(0.1, 0.9, txt, ha='left', va='top', fontsize=14,
                 transform=ax3.transAxes)
        txt = "Method = {0:s}".format(self.method)
        txt += "\n" + r"$\phi(0) = {0:0.4f}$".format(self.phi0)
        txt += "\n" + r"$\dot\phi(0) = {0:0.4f}$".format(self.vphi0)
        ax3.text(0.9, 0.9, txt, ha='right', va='top', fontsize=14,
                 transform=ax3.transAxes)

        fig.tight_layout()
        fig.savefig("ode_pend.pdf")
        plt.show()

if __name__ == "__main__":
    # Example :
    p = Pendulum()
    p.solve()
    p.plot()



