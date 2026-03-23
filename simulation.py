"""
simulation.py — v6 patched.

Fixes in this version
_____________________
• GasPlanet: redshift + fade applies correctly while spaghettifying,
  because CelestialBody.update() now computes zf from NEAREST BH distance.
  GasPlanet now also tints itself red proportionally to spaghettification
  intensity, visible long before the event horizon.
• GasPlanet._emit_particles(): particles inherit orbital tangential velocity
  so they stream along the planet's previous orbit (comet-tail behaviour)
  instead of scattering isotropically.
"""
import numpy as np
import math
from physics import Physics


class AccretionDisk:
    """Keplerian thin disk centred on a BlackHoleBody."""

    def __init__(self, bh, n: int = 3200, physics: Physics = None):
        self.bh         = bh
        self.n          = n
        self._phys      = physics or Physics()
        self._phys.update_properties(bh.mass, bh.spin)
        self._last_mass = bh.mass
        self.rebuild()

    def rebuild(self):
        bh = self.bh
        from config import Config
        self.n = max(0, getattr(Config, 'DISK_PARTICLE_COUNT', 3200))
        if self.n == 0: return
        rng   = np.random.default_rng(1337 + (bh.id if hasattr(bh, 'id') else 0))
        r_in  = bh.r_isco_sim * 1.01
        r_out = bh.r_isco_sim * 8.0
        u           = rng.uniform(0.0, 1.0, self.n) ** 1.30
        self._u     = u.copy()
        r_sim       = r_in + (r_out - r_in) * u
        self.r_sim  = r_sim
        self.r_px   = r_sim * bh.SIM_SCALE
        self.theta  = rng.uniform(0.0, 2*np.pi, self.n)
        self.y_off_local = rng.normal(0.0, self.r_px * 0.030)
        self._compute_dynamics()

    def _compute_dynamics(self):
        bh  = self.bh
        GM  = bh.GM
        self.omega  = np.sqrt(GM) * self.r_sim**(-1.5)
        v_isco = np.sqrt(GM / (bh.r_isco_sim * max(1.0 - 1.5/max(bh.r_isco_sim, 0.0001), 0.01)))
        self.v_frac = (self.r_sim * self.omega) / max(v_isco, 1e-9) * 0.42
        T                = self._phys.disk_temperature(self.r_sim)
        self.base_colors = self._phys.temperature_to_rgb(T)
        self.T           = T
        self.gz          = self._phys.grav_redshift(self.r_sim)
        self.alpha       = np.clip(T * 3.2 + 0.04, 0.04, 1.0).astype(float)

    def resync(self):
        bh = self.bh
        self._phys.update_properties(bh.mass, bh.spin)
        r_in  = bh.r_isco_sim * 1.01
        r_out = bh.r_isco_sim * 8.0
        self.r_sim = r_in + (r_out - r_in) * self._u
        self.r_px  = self.r_sim * bh.SIM_SCALE
        self._compute_dynamics()
        self._last_mass = bh.mass

    def update(self, dt: float, speed: float = 6.5):
        bh = self.bh
        if abs(bh.mass - self._last_mass) > 5e-4:
            self.resync()
        fd_omega = (2 * bh.GM * (bh.spin * bh.GM)) / (self.r_sim**3 + 1e-5)
        self.theta += (self.omega + fd_omega) * dt * speed

    def positions_3d(self) -> np.ndarray:
        bh = self.bh
        x  = self.r_px * np.cos(self.theta) + bh.pos[0]
        z  = self.r_px * np.sin(self.theta) + bh.pos[2]
        y  = self.y_off_local               + bh.pos[1]
        return np.stack([x, y, z], axis=1)

    def colors_frame(self, camera) -> np.ndarray:
        bh      = self.bh
        cam     = camera.position
        cam_xz  = np.array([cam[0] - bh.pos[0], cam[2] - bh.pos[2]])
        n_cam   = np.linalg.norm(cam_xz)
        if n_cam > 0: cam_xz /= n_cam
        beta = (-np.sin(self.theta)*cam_xz[0] + np.cos(self.theta)*cam_xz[1]) * self.v_frac
        c    = self._phys.doppler_shift(self.base_colors.copy(), beta)
        from config import Config
        br   = Config.DISK_BRIGHTNESS * Config.GLOBAL_BRIGHTNESS
        c    = np.clip(c.astype(float) * self.gz[:, None] * br, 0, 255).astype(np.uint8)
        return c


#_____________________________________________________________________________# 


class FreeParticles:
    """3D Cartesian RK4 integration under multi-BH gravity."""

    def __init__(self, physics: Physics, n: int = 360):
        self.phys = physics
        self.n    = n
        rng       = np.random.default_rng(7777)

        r_range = np.linspace(3.5, 14.0, n)
        phi_rng = rng.uniform(0.0, 2*np.pi, n)
        tilt    = rng.uniform(-np.pi/5, np.pi/5, n)
        rx = r_range * np.cos(phi_rng) * np.cos(tilt)
        ry = r_range * np.sin(tilt)
        rz = r_range * np.sin(phi_rng) * np.cos(tilt)
        self.pos = np.stack([rx, ry, rz], axis=1)

        GM     = physics.GM
        v_circ = np.sqrt(GM / np.maximum(r_range, 0.52))
        tx = -np.sin(phi_rng) * (1.0 + rng.normal(0, 0.08, n))
        tz =  np.cos(phi_rng) * (1.0 + rng.normal(0, 0.08, n))
        ty =  rng.normal(0, 0.05, n)
        t_len = np.sqrt(tx**2 + ty**2 + tz**2)
        tx /= t_len;  ty /= t_len;  tz /= t_len
        self.vel = np.stack([tx, ty, tz], axis=1) * v_circ[:, None]

        t   = rng.choice(4, n, p=[0.35, 0.30, 0.20, 0.15])
        col = np.zeros((n, 3), np.uint8)
        col[t==0] = [120, 190, 255]
        col[t==1] = [255, 165,  55]
        col[t==2] = [255, 240, 160]
        col[t==3] = [160, 255, 200]
        noise = rng.integers(-30, 30, (n, 3))
        self.colors = np.clip(col.astype(int) + noise, 0, 255).astype(np.uint8)
        self.size   = rng.uniform(0.8, 2.4, n)
        self.alpha  = np.full(n, 0.75)

    def update(self, dt: float, bh_list: list, speed: float = 5.0):
        if self.n == 0: return
        dt_sim = dt * speed
        N_sub  = 4
        ds     = dt_sim / N_sub

        for _ in range(N_sub):
            a1 = Physics.multi_bh_accel(self.pos, bh_list)
            p2 = self.pos + 0.5*ds*self.vel;  v2 = self.vel + 0.5*ds*a1
            a2 = Physics.multi_bh_accel(p2, bh_list)
            p3 = self.pos + 0.5*ds*v2;        v3 = self.vel + 0.5*ds*a2
            a3 = Physics.multi_bh_accel(p3, bh_list)
            p4 = self.pos +    ds*v3;          v4 = self.vel +    ds*a3
            a4 = Physics.multi_bh_accel(p4, bh_list)
            self.pos += (ds/6.0) * (self.vel + 2*v2 + 2*v3 + v4)
            self.vel += (ds/6.0) * (a1 + 2*a2 + 2*a3 + a4)

        remove = np.zeros(self.n, bool)
        for bh in bh_list:
            if not bh.active: continue
            bh_sim    = bh.pos / bh.SIM_SCALE
            dist      = np.linalg.norm(self.pos - bh_sim, axis=1)
            dead_mask = dist < bh.rs_sim * 1.04
            c = int(dead_mask.sum())
            if c > 0:
                from config import Config
                bh.accrete(c * getattr(Config, 'ACCRETION_DM_PARTICLE', 0.00005))
            remove |= dead_mask
            remove |= dist > 80.0

        if remove.any():
            keep = ~remove
            self.pos    = self.pos[keep]
            self.vel    = self.vel[keep]
            self.colors = self.colors[keep]
            self.size   = self.size[keep]
            self.alpha  = self.alpha[keep]
            self.n      = int(keep.sum())

    def absorb_disk(self, disk):
        n    = disk.n
        if n == 0: return
        bh   = disk.bh
        S    = bh.SIM_SCALE
        pos3 = disk.positions_3d() / S
        vt   = disk.r_sim * disk.omega
        tx   = -np.sin(disk.theta);  tz = np.cos(disk.theta)
        vel3 = np.stack([tx*vt, np.zeros(n), tz*vt], axis=1)
        self.pos    = np.vstack([self.pos, pos3])
        self.vel    = np.vstack([self.vel, vel3])
        self.colors = np.vstack([self.colors, disk.base_colors])
        self.size   = np.append(self.size, np.ones(n)*1.5)
        self.alpha  = np.append(self.alpha, disk.alpha)
        self.n     += n
        disk.n      = 0

    def positions_3d(self) -> np.ndarray:
        return self.pos * self.phys.SIM_SCALE

    def colors_frame(self) -> np.ndarray:
        from config import Config
        r   = np.linalg.norm(self.pos, axis=1)
        rs  = self.phys.rs_sim
        gz  = np.sqrt(np.maximum(0.0, 1.0 - rs / np.maximum(r, rs*1.01)))
        c   = self.colors.astype(float) * gz[:, None] * getattr(Config, 'GLOBAL_BRIGHTNESS', 1.0)
        if getattr(Config, 'PARTICLE_TEMP_GLOW', True):
            v2   = np.sum(self.vel**2, axis=1)
            heat = np.clip(v2 * 0.08, 0.0, 1.0)
            glow = np.array([210, 240, 255], float) * getattr(Config, 'GLOBAL_BRIGHTNESS', 1.0)
            c    = c*(1.0-heat[:,None]) + glow*heat[:,None]
        return np.clip(c, 0, 255).astype(np.uint8)


#________________________________________________________________________#


class CelestialBody:
    def __init__(self, physics, pos, radius, color, vel=(0,0,0),
                 cohesion=1.0, respawn=False):
        self.phys           = physics
        self.initial_pos    = np.array(pos,  dtype=float)
        self.initial_vel    = np.array(vel,  dtype=float)
        self.initial_radius = radius
        self.cohesion       = cohesion
        self.respawn        = respawn
        self.pos            = self.initial_pos.copy()
        self.r_px           = radius * physics.SIM_SCALE
        self.base_color     = np.array([color[0], color[1], color[2]], float)
        self.color          = (int(color[0]), int(color[1]), int(color[2]), 255)
        self.vel            = self.initial_vel.copy()
        self.active         = True

    def _do_respawn(self):
        self.pos    = self.initial_pos.copy()
        self.vel    = self.initial_vel.copy()
        self.r_px   = self.initial_radius * self.phys.SIM_SCALE
        self.color  = (int(self.base_color[0]), int(self.base_color[1]),
                       int(self.base_color[2]), 255)
        self.active = True

    def update(self, dt, speed=5.0, free_system=None, bh_list=None):
        if not self.active: return
        from config import Config

        dt_sim  = dt * speed
        pos_sim = self.pos / self.phys.SIM_SCALE

        #Find nearest active BH
        nearest_bh = None;  nearest_r = 1e18
        if bh_list:
            for bh in bh_list:
                if not bh.active: continue
                bh_sim = bh.pos / bh.SIM_SCALE
                d = np.linalg.norm(pos_sim - bh_sim)
                if d < nearest_r:
                    nearest_r = d;  nearest_bh = bh
        rs = nearest_bh.rs_sim if nearest_bh else self.phys.rs_sim

        #Time dilation
        if getattr(Config, 'ENABLE_TIME_DILATION_OBJECTS', True):
            tf = math.sqrt(max(0.0001, 1.0 - rs / max(nearest_r, rs*1.001)))
            dt_sim *= tf

        #Redshift fading (applied always, based on nearest BH distance)
        B = getattr(Config, 'GLOBAL_BRIGHTNESS', 1.0)
        if getattr(Config, 'ENABLE_REDSHIFT_FADING', True):
            zf    = math.sqrt(max(0.0, 1.0 - rs / max(nearest_r, rs*1.001)))
            alpha = int(np.clip(zf**0.5 * 255, 0, 255))
            self.color = (int(np.clip(self.base_color[0]*zf**0.5*B, 0, 255)),
                          int(np.clip(self.base_color[1]*zf**1.5*B, 0, 255)),
                          int(np.clip(self.base_color[2]*zf**2.5*B, 0, 255)),
                          alpha)
        else:
            self.color = (int(np.clip(self.base_color[0]*B, 0, 255)),
                          int(np.clip(self.base_color[1]*B, 0, 255)),
                          int(np.clip(self.base_color[2]*B, 0, 255)), 255)

        #Inside event horizon: accelerated fade → deactivate
        if nearest_bh and nearest_r <= rs * 1.08:
            if getattr(Config, 'ENABLE_REDSHIFT_FADING', True):
                alpha = int(max(0, self.color[3] - 20*speed))
                self.color = (self.color[0]*alpha//255, self.color[1]*alpha//255,
                              self.color[2]*alpha//255, alpha)
                if nearest_r <= rs*1.01 or alpha == 0:
                    if self.respawn: self._do_respawn()
                    else:
                        self.active = False
                        if nearest_bh:
                            nearest_bh.accrete(getattr(Config, 'ACCRETION_DM_BODY', 0.005))
            else:
                self.r_px *= 0.85
                if self.r_px < 1.0 or nearest_r <= rs*1.01:
                    if self.respawn: self._do_respawn()
                    else:
                        self.active = False
                        if nearest_bh:
                            nearest_bh.accrete(getattr(Config, 'ACCRETION_DM_BODY', 0.005))
            if not self.active: return

        #Multi-BH acceleration
        if bh_list:
            accel = Physics.multi_bh_accel_single(pos_sim, bh_list)
        else:
            direction = -pos_sim / max(np.linalg.norm(pos_sim), 1e-5)
            accel = direction * self.phys.GM / max(nearest_r - rs, rs*0.05)**2

        self.vel  += accel * dt_sim
        pos_sim   += self.vel * dt_sim
        self.pos   = pos_sim * self.phys.SIM_SCALE

    def get_silhouette_points(self, camera, n_points=40):
        if not self.active: return None
        d    = self.pos - camera.position
        dist = np.linalg.norm(d)
        if dist < 1e-5: return None
        fwd  = d / dist
        up   = np.array([0.0, 1.0, 0.0])
        if abs(np.dot(fwd, up)) > 0.99:
            up = np.array([1.0, 0.0, 0.0])
        right  = np.cross(fwd, up);  right /= np.linalg.norm(right)
        up_cam = np.cross(right, fwd)
        theta  = np.linspace(0, 2*np.pi, n_points, endpoint=False)
        return (self.pos[None,:]
                + (right[None,:]*np.cos(theta)[:,None]
                   + up_cam[None,:]*np.sin(theta)[:,None]) * self.r_px)


#_____________________________________________________________________________#


class GasPlanet(CelestialBody):
    """
    GasPlanet extends CelestialBody with tidal spaghettification.

    Fixes vs previous version
    _________________________
    • Roche limit check uses distance from NEAREST BH, not from origin.
      Correct for multi-BH and for BHs that have drifted away from origin.
    • Spaghettification intensity drives an additional red tint blended into
      self.color, so the planet visibly reddens while being stretched —
      long before it reaches the event horizon where the normal redshift
      fade kicks in.
    • _emit_particles(): particles receive the planet's full orbital velocity
      (tangential component) so they stream along the previous orbital path
      (comet-tail / TDE debris stream) instead of scattering isotropically.
    """

    def update(self, dt, speed=5.0, free_system=None, bh_list=None):
        if not self.active: return

        #CelestialBody handles: gravity, time dilation, redshift, EH fade
        super().update(dt, speed, free_system, bh_list)
        if not self.active: return

        pos_sim = self.pos / self.phys.SIM_SCALE

        #--- Nearest BH for Roche limit (FIX: use BH position, not origin) ---
        nearest_bh = None;  nearest_r = 1e18
        if bh_list:
            for bh in bh_list:
                if not bh.active: continue
                bh_sim = bh.pos / bh.SIM_SCALE
                d = np.linalg.norm(pos_sim - bh_sim)
                if d < nearest_r:
                    nearest_r = d;  nearest_bh = bh

        #Fall back to distance from origin when no BH list provided
        r_from_bh = nearest_r if nearest_bh else np.linalg.norm(pos_sim)

        from config import Config
        base_roche = getattr(Config, 'ROCHE_LIMIT_BASE', 15.0)
        cohesion   = getattr(Config, 'PLANET_COHESION', 1.0) * self.cohesion
        safe_r     = base_roche * max(0.1, cohesion)

        if r_from_bh < safe_r and self.r_px > 0.5:
            intensity = np.clip((safe_r - r_from_bh) / safe_r, 0.0, 1.0)

            #Shrink
            self.r_px = max(0.0, self.r_px - (intensity**2)*8.0*dt*speed)

            #Spaghettification red tint
            #Blend current color toward hot red as intensity increases.
            #Gives a dramatic visual cue while the planet is being torn apart.
            if getattr(Config, 'ENABLE_REDSHIFT_FADING', True):
                r_tint = int(np.clip(255 * intensity, 0, 255))
                g_tint = int(np.clip(self.color[1] * (1.0 - intensity * 0.8), 0, 255))
                b_tint = int(np.clip(self.color[2] * (1.0 - intensity), 0, 255))
                #Blend tint into existing (redshifted) color
                blend  = min(intensity * 1.5, 1.0)
                cr     = int(self.color[0]*(1-blend) + r_tint*blend)
                cg     = int(self.color[1]*(1-blend) + g_tint*blend)
                cb     = int(self.color[2]*(1-blend) + b_tint*blend)
                self.color = (
                    int(np.clip(cr, 0, 255)),
                    int(np.clip(cg, 0, 255)),
                    int(np.clip(cb, 0, 255)),
                    self.color[3],   #keep alpha from CelestialBody fade
                )

            #Particle emission
            emission_r = getattr(Config, 'GAS_EMISSION_RATE', 1.0)
            if free_system is not None and np.random.rand() < intensity*1.5*emission_r:
                amount = int(intensity*10*emission_r) + 1
                self._emit_particles(
                    free_system, amount, pos_sim, nearest_bh)

    def _emit_particles(self, free, amount, pos_sim, nearest_bh=None):
        """
        Emit `amount` particles at the planet surface.

        Velocity fix
        ____________
        Particles receive the planet's orbital tangential velocity so they
        continue along the previous orbital path (like a TDE debris stream /
        comet tail).  A small random scatter and a radial-infall component are
        added for realism, but the orbital tangent dominates.

        Steps:
          1. Compute radial direction from nearest BH to planet.
          2. Tangential direction = perpendicular to radial in XZ plane.
          3. Project self.vel onto tangent → v_tang (signed orbital speed).
          4. Each particle starts with v_tang * tangent + small noise.
          5. Scale by PARTICLE_PLUNGE_FACTOR.
        """
        from config import Config
        rng = np.random.default_rng()

        #--- Surface scatter positions ---
        u   = rng.uniform(-1, 1, amount)
        phi = rng.uniform(0, 2*np.pi, amount)
        st  = np.sqrt(np.maximum(0.0, 1 - u**2))
        rs  = max(0.01, self.r_px / self.phys.SIM_SCALE)
        pos3 = np.stack([pos_sim[0] + rs*st*np.cos(phi),
                          pos_sim[1] + rs*u,
                          pos_sim[2] + rs*st*np.sin(phi)], axis=1)   #(N,3) sim units

        #--- Orbital tangential velocity ---
        #Radial direction from BH (or origin) to planet
        if nearest_bh is not None:
            bh_sim  = nearest_bh.pos / nearest_bh.SIM_SCALE
            radial  = pos_sim - bh_sim
        else:
            radial  = pos_sim.copy()

        radial_xz_len = math.sqrt(radial[0]**2 + radial[2]**2)
        if radial_xz_len > 1e-6:
            #Tangential direction in XZ plane (90° counter-clockwise from radial)
            tang = np.array([-radial[2], 0.0, radial[0]], float) / radial_xz_len
        else:
            tang = np.array([1.0, 0.0, 0.0], float)

        #Project planet velocity onto tangent to get orbital speed
        v_orbital = float(np.dot(self.vel, tang))   #signed scalar (sim units/s)

        #Each particle gets: orbital tangent velocity + small noise
        noise_scale = getattr(Config, 'PARTICLE_DYNAMICS_NOISE', 0.04)
        vel3  = tang[None, :] * v_orbital            #(N,3): purely tangential
        vel3  = vel3 + rng.normal(0, noise_scale, (amount, 3))

        #Optionally add a small radial infall component (pulled toward BH)
        if radial_xz_len > 1e-6:
            infall_dir  = -radial / (np.linalg.norm(radial) + 1e-9)
            infall_speed = abs(v_orbital) * 0.15       #15 % of orbital speed
            vel3 += infall_dir[None, :] * infall_speed

        vel3 *= getattr(Config, 'PARTICLE_PLUNGE_FACTOR', 1.0)

        #--- Append to FreeParticles ---
        free.pos    = np.vstack([free.pos, pos3])
        free.vel    = np.vstack([free.vel, vel3])
        colors = np.clip(
            np.full((amount, 3), self.base_color) + rng.integers(-20, 20, (amount, 3)),
            0, 255).astype(np.uint8)
        free.colors = np.vstack([free.colors, colors])
        free.size   = np.append(free.size, rng.uniform(0.5, 1.5, amount))
        free.alpha  = np.append(free.alpha, np.ones(amount) * 0.75)
        free.n     += amount
