"""
renderer.py — v7

- No explicit black disk for BH shadow.
- Event Horizon (photon ring) is the only BH visual boundary.
- Soft gradient shadow applied via numpy buf — no hard circle masking.
- Realistic 3D star sphere when Config.REALISTIC_STARS = True.
"""
import math
import numpy as np
import pygame

_TWO_PI = 2.0 * math.pi
_STAR_SPHERE_R = 80_000.0


class Renderer:
    def __init__(self, W: int, H: int, physics):
        self.W    = W
        self.H    = H
        self.phys = physics

        pygame.font.init()
        self.font     = pygame.font.SysFont("monospace", 12)
        self.font_med = pygame.font.SysFont("monospace", 14)

        self.N_ang = 720

        from config import Config
        self.rebuild_stars(getattr(Config, 'STAR_COUNT', 3400))

    #fake stars

    def rebuild_stars(self, NS):
        NS  = max(0, int(NS))
        rng = np.random.default_rng(99 + NS)

        # Colours
        b    = np.clip(rng.exponential(0.18, NS), 0.0, 1.0)
        tp   = rng.choice(3, NS, p=[0.54, 0.28, 0.18])
        base = np.zeros((NS, 3), float)
        base[tp==0] = [255, 255, 255]
        base[tp==1] = [155, 200, 255]
        base[tp==2] = [255, 240, 165]
        self.s_raw = np.clip(base * b[:,None], 0, 255).astype(np.uint8)

        # 3-D world positions (fixed on a large sphere)
        #Used when REALISTIC_STARS = True.
        cos_t = rng.uniform(-1.0, 1.0, NS)
        phi_s = rng.uniform(0.0, _TWO_PI, NS)
        sin_t = np.sqrt(np.maximum(0.0, 1.0 - cos_t**2))
        R     = _STAR_SPHERE_R
        self.s_pos3d = np.stack([
            R * sin_t * np.cos(phi_s),
            R * cos_t,
            R * sin_t * np.sin(phi_s),
        ], axis=1).astype(float)          #(NS, 3) world pixels

        # 2-D screen positions (original camera-fixed mode)
        self.s_sx = rng.integers(0, self.W, NS).astype(float)
        self.s_sy = rng.integers(0, self.H, NS).astype(float)

    # Fast scatter

    def _scatter(self, buf, xs, ys, cols):
        x = xs.astype(int);  y = ys.astype(int)
        v = (x>=0)&(x<self.W)&(y>=0)&(y<self.H)
        if not v.any(): return
        np.add.at(buf, (x[v], y[v]), cols[v].astype(np.int16))

    # Stars

    def _draw_stars(self, buf, camera, bh_projs):
        from config import Config
        B  = getattr(Config, 'GLOBAL_BRIGHTNESS', 1.0)

        if getattr(Config, 'REALISTIC_STARS', True):
            # 3D mode: project star sphere onto screen
            sx, sy, depths, fwd = camera.project_batch(self.s_pos3d, self.W, self.H)
            #Only stars in front of the camera
            vis  = fwd
            lx   = sx[vis].copy();  ly = sy[vis].copy()
            raw  = self.s_raw[vis]

            #Apply thin-lens deflection from each BH
            for bhx, bhy, theta_E, _bd, _sr, _bh in bh_projs:
                if theta_E < 1.0: continue
                dx = lx-bhx;  dy = ly-bhy
                β  = np.maximum(np.sqrt(dx*dx+dy*dy), 0.5)
                nx = dx/β;    ny = dy/β
                θp = (β + np.sqrt(β*β + 4*theta_E*theta_E))*0.5
                lx = bhx + nx*θp
                ly = bhy + ny*θp

            inb = (lx>=0)&(lx<self.W)&(ly>=0)&(ly<self.H)
            c   = np.clip(raw[inb].astype(float) * B, 0, 255).astype(np.int16)
            np.add.at(buf, (lx[inb].astype(int), ly[inb].astype(int)), c)

        else:
            # 2D mode: camera-fixed positions (original behaviour)
            lx = self.s_sx.copy();  ly = self.s_sy.copy()
            for bhx, bhy, theta_E, _bd, _sr, _bh in bh_projs:
                if theta_E < 1.0: continue
                dx = lx-bhx;  dy = ly-bhy
                β  = np.maximum(np.sqrt(dx*dx+dy*dy), 0.5)
                nx = dx/β;    ny = dy/β
                θp = (β + np.sqrt(β*β + 4*theta_E*theta_E))*0.5
                lx = bhx + nx*θp
                ly = bhy + ny*θp
            inb = (lx>=0)&(lx<self.W)&(ly>=0)&(ly<self.H)
            c   = np.clip(self.s_raw[inb].astype(float)*B, 0, 255).astype(np.int16)
            np.add.at(buf, (lx[inb].astype(int), ly[inb].astype(int)), c)

    # Particles

    def _draw_particles(self, buf, camera, disk_list, free, side, bh_projs):
        from config import Config
        show_disk = getattr(Config, 'USE_VIRTUAL_ACCRETION_DISK', True)
        ref_depth = bh_projs[0][3] if bh_projs else 99999

        passes = []
        if show_disk:
            for disk in disk_list:
                if disk.bh.active and disk.n > 0:
                    passes.append((disk.positions_3d(),
                                   disk.colors_frame(camera),
                                   disk.alpha))
        if free.n > 0:
            passes.append((free.positions_3d(),
                           free.colors_frame(),
                           np.full(free.n, 0.75)))

        for pos3, cols, alpha in passes:
            if len(pos3) == 0: continue
            sx, sy, depths, fwd = camera.project_batch(pos3, self.W, self.H)

            mask = fwd & (depths >= ref_depth if side=='far' else depths < ref_depth)
            idx  = np.where(mask)[0]
            if not len(idx): continue

            lx = sx[idx].copy();  ly = sy[idx].copy()

            if side == 'far':
                for bhx, bhy, theta_E, bh_depth, _sr, _bh in bh_projs:
                    if theta_E < 0.5: continue
                    behind = depths[idx] > bh_depth
                    if not behind.any(): continue
                    dx = lx[behind]-bhx;  dy = ly[behind]-bhy
                    β  = np.maximum(np.sqrt(dx*dx+dy*dy), 0.5)
                    nx = dx/β;  ny = dy/β
                    θp = (β + np.sqrt(β*β + 4*theta_E*theta_E))*0.5
                    lx[behind] = bhx + nx*θp
                    ly[behind] = bhy + ny*θp
            else:
                #Mask near-side particles inside any BH shadow
                valid = np.ones(len(idx), bool)
                for bhx, bhy, _te, _bd, shadow_r_px, _bh in bh_projs:
                    if shadow_r_px <= 0: continue
                    dist2 = (lx-bhx)**2 + (ly-bhy)**2
                    valid &= dist2 > (shadow_r_px * 0.85)**2
                lx = lx[valid];  ly = ly[valid]
                idx = idx[valid]
                if not len(idx): continue

            a = alpha[idx]
            c = (cols[idx].astype(float) * a[:,None]).astype(np.uint8)
            self._scatter(buf, lx, ly, c)

            bright = a > 0.40
            if bright.any():
                gc = (c[bright].astype(float)*0.25).astype(np.uint8)
                bx = lx[bright];  by = ly[bright]
                for ddx,ddy in ((-1,0),(1,0),(0,-1),(0,1)):
                    self._scatter(buf, bx+ddx, by+ddy, gc)

    # Photon ring

    def _draw_photon_ring(self, buf, camera, disk, bhx, bhy, bh_depth, shadow_r_px):
        N   = self.N_ang
        pos = disk.positions_3d()
        col = disk.colors_frame(camera)
        alp = disk.alpha

        sx, sy, depths, fwd = camera.project_batch(pos, self.W, self.H)
        valid = np.where(fwd)[0]
        if not len(valid): return

        dx  = sx[valid]-bhx;  dy = sy[valid]-bhy
        phi = np.arctan2(dy, dx) % _TWO_PI
        far  = depths[valid] >= bh_depth
        near = ~far

        ring_col = np.zeros((N,3), float)
        def accum(mask, phi_vals, weight, phi_offset=0.0):
            if not mask.any(): return
            idm  = valid[mask]
            φ    = (phi_vals[mask]+phi_offset) % _TWO_PI
            bins = (φ/_TWO_PI*N).astype(int) % N
            w    = alp[idm]*weight
            c    = col[idm].astype(float)
            for ch in range(3):
                np.add.at(ring_col[:,ch], bins, w*c[:,ch])

        accum(far,  phi, 3.2, 0.0)
        accum(near, phi, 0.6, math.pi*0.90)
        mx = ring_col.max()
        if mx < 1e-6: return
        ring_col = np.clip(ring_col/mx*220.0, 0, 255)

        ang = np.linspace(0, _TWO_PI, N, endpoint=False)
        ca  = np.cos(ang);  sa = np.sin(ang)
        for rr, wt in ((shadow_r_px-.5, 0.55),(shadow_r_px+.5, 1.00),(shadow_r_px+1.5, 0.40)):
            rx = (bhx+rr*ca).astype(int);  ry = (bhy+rr*sa).astype(int)
            v  = (rx>=0)&(rx<self.W)&(ry>=0)&(ry<self.H)
            if v.any():
                np.add.at(buf, (rx[v],ry[v]), (ring_col[v]*wt).astype(np.int16))

    # Celestial bodies

    def _draw_celestial_body(self, surface, camera, obj, bh_projs, side):
        pts_3d = obj.get_silhouette_points(camera, n_points=50)
        if pts_3d is None: return
        c_proj = camera.project_single(obj.pos, self.W, self.H)
        if c_proj is None: return
        _, _, obj_depth = c_proj

        if bh_projs:
            bh_depth = bh_projs[0][3]
            shadow_r = bh_projs[0][4]
            is_near  = obj_depth < (bh_depth - shadow_r * 1.5)
        else:
            is_near = True

        if (side=='near' and not is_near) or (side=='far' and is_near):
            return

        sx, sy, depths, fwd_mask = camera.project_batch(pts_3d, self.W, self.H)
        if not fwd_mask.all(): return

        if bh_projs and side == 'far':
            bhx, bhy, theta_E, bh_depth, shadow_r_px, _bh = bh_projs[0]
            if bh_depth < 99999 and obj_depth > bh_depth and theta_E > 1e-4:
                d_ls = obj_depth - bh_depth
                d_os = max(obj_depth, 0.01)
                lthE_weak  = theta_E * math.sqrt(d_ls / d_os)
                rs_px      = shadow_r_px / 2.598
                strong_boost = rs_px / max(d_ls, rs_px * 0.25)
                lthE = lthE_weak * (1.0 + 1.8 * strong_boost)
                if lthE > 1e-4:
                    p1x,p1y,p2x,p2y,_,_,_ = self.phys.lens_all(
                        sx, sy, bhx, bhy, lthE, shadow_r_px)
                    for px_arr, py_arr in ((p1x,p1y),(p2x,p2y)):
                        pts = np.vstack([np.column_stack((px_arr,py_arr)),
                                         np.column_stack((px_arr,py_arr))[:1]])
                        xf, yf = self.phys.CubicSpline(pts)
                        self._draw_alpha_polygon(surface, obj.color, list(zip(xf,yf)))
                    return

        pts = np.vstack([np.column_stack((sx,sy)), np.column_stack((sx,sy))[:1]])
        xf, yf = self.phys.CubicSpline(pts)
        self._draw_alpha_polygon(surface, obj.color, list(zip(xf,yf)))

    def _draw_alpha_polygon(self, surface, color, points):
        if not points: return
        if len(color) < 4 or color[3] == 255:
            pygame.draw.polygon(surface, color[:3], points)
            return
        if color[3] == 0: return
        min_x = int(min(p[0] for p in points));  max_x = int(max(p[0] for p in points))
        min_y = int(min(p[1] for p in points));  max_y = int(max(p[1] for p in points))
        w = max_x-min_x+6;  h = max_y-min_y+6
        if 0 < w < self.W and 0 < h < self.H:
            ov = pygame.Surface((w,h), pygame.SRCALPHA)
            pygame.draw.polygon(ov, color, [(p[0]-min_x+3,p[1]-min_y+3) for p in points])
            surface.blit(ov, (min_x-3, min_y-3))

    # Merger flash

    def _draw_merger_flash(self, surface, bh, camera):
        res = camera.project_single(bh.pos, self.W, self.H)
        if res is None: return
        bhx, bhy, bh_depth = res
        r_flash = camera.get_screen_radius(
            bh.shadow_r_sim*bh.SIM_SCALE*4.0, bh_depth, self.W, self.H)
        alpha = int(bh.merge_flash/1.5 * 200)
        if r_flash < 1: return
        fs = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for dr in range(5):
            pygame.draw.circle(fs, (255,200,100,max(0,alpha-dr*40)),
                               (int(bhx),int(bhy)), int(r_flash)+dr, 2)
        surface.blit(fs, (0,0))

    # Bloom

    def _bloom(self, surface):
        W,H = self.W,self.H
        sm  = pygame.transform.smoothscale(surface,(W//4,H//4))
        bl  = pygame.transform.smoothscale(sm,(W,H));  bl.set_alpha(76)
        surface.blit(bl,(0,0),special_flags=pygame.BLEND_ADD)
        sm2 = pygame.transform.smoothscale(surface,(W//2,H//2))
        bl2 = pygame.transform.smoothscale(sm2,(W,H));  bl2.set_alpha(22)
        surface.blit(bl2,(0,0),special_flags=pygame.BLEND_ADD)

    # Config panel

    def draw_config_panel(self, surface, panel_state):
        if not panel_state.get('visible'): return
        from config import Config
        params  = panel_state['params']
        cursor  = panel_state['cursor']
        PW, PH  = 380, min(len(params)*18+50, self.H-20)
        px0, py0 = self.W-PW-8, 8

        overlay = pygame.Surface((PW,PH), pygame.SRCALPHA)
        overlay.fill((10,10,30,210))
        surface.blit(overlay, (px0,py0))
        pygame.draw.rect(surface, (80,120,255), (px0,py0,PW,PH), 1)

        title = self.font_med.render("CONFIG  (Tab=close  ↑↓=select  ←→=change)",
                                     True, (180,220,255))
        surface.blit(title, (px0+6, py0+4))

        for i, p in enumerate(params):
            y   = py0 + 22 + i*18
            val = getattr(Config, p['attr'], '?')
            if isinstance(val, bool):
                val_str = "ON " if val else "OFF"
                col_v   = (100,255,140) if val else (255,100,100)
            elif isinstance(val, float):
                val_str = f"{val:.4f}" if val < 0.01 else f"{val:.3f}"
                col_v   = (255,240,160)
            elif isinstance(val, int):
                val_str = str(val);  col_v = (255,240,160)
            else:
                val_str = str(val);  col_v = (200,200,255)

            if i == cursor:
                hl = pygame.Surface((PW-2,17), pygame.SRCALPHA)
                hl.fill((40,60,120,180))
                surface.blit(hl, (px0+1, y-1))
            arrow = "▶ " if i==cursor else "  "
            label = self.font.render(f"{arrow}{p['name']:<28}", True,
                                     (255,255,255) if i==cursor else (180,180,200))
            value = self.font.render(val_str, True, col_v)
            surface.blit(label, (px0+4, y))
            surface.blit(value, (px0+PW-65, y))

        hint = self.font.render("H=BH  B=planet  V=body  O=pause  ESC=quit",
                                True, (120,140,180))
        surface.blit(hint, (px0+4, py0+PH-14))

    def _hud(self, surface, fps, bh_list, show_particles, selected_bh_idx):
        from config import Config
        active = [b for b in bh_list if b.active]
        bh_str = "  ".join(
            f"{'[►]' if i==selected_bh_idx else '   '}{b.label} M={b.mass:.2f} a={b.spin:.2f}"
            for i,b in enumerate(bh_list) if b.active)
        lines = [
            (f"FPS {fps:3.0f} | BHs:{len(active)} | "
             f"Time×{Config.TIME_LAPSE:.1f} | "
             f"{'3D★' if Config.REALISTIC_STARS else '2D★'} | "
             f"Phys:{Config.PHYSICS_MODE} | "
             f"Disk:{'ON' if Config.USE_VIRTUAL_ACCRETION_DISK else 'OFF'} | "
             f"Part:{'ON' if show_particles else 'OFF'} | Tab=cfg"),
            bh_str,
        ]
        legend = self.font.render(
            "Units: Mass [M]  Distance [R_s]  Velocity [c]", True, (150,160,180))
        surface.blit(legend, (10,10))
        for i, line in enumerate(lines):
            s = self.font.render(line, True, (200,220,255))
            surface.blit(s, (10, self.H - 14*(len(lines)-i)))

    # Main render

    def render(self, surface, camera, disk_list, free, objects,
               bh_list, fps, show_particles=True,
               selected_bh_idx=0, panel_state=None):

        buf = np.zeros((self.W, self.H, 3), dtype=np.int16)
        from config import Config

        #Project all active BHs
        bh_projs = []
        for bh in bh_list:
            if not bh.active: continue
            res = camera.project_single(bh.pos, self.W, self.H)
            if res is None: continue
            bhx, bhy, bh_depth = res
            dist_fc    = math.hypot(bhx-self.W/2, bhy-self.H/2)
            max_r      = max(self.W, self.H)*0.7
            dist_fade  = np.clip(1.0-(dist_fc-max_r)/max_r, 0.0, 1.0)
            depth_fade = np.clip((bh_depth-0.1)/5.0, 0.0, 1.0)
            fade       = dist_fade * depth_fade
            shadow_r_px   = camera.get_screen_radius(
                bh.shadow_r_sim*bh.SIM_SCALE, bh_depth, self.W, self.H) * fade
            einstein_r_px = camera.get_screen_radius(
                bh.einstein_r_sim*bh.SIM_SCALE, bh_depth, self.W, self.H) * fade
            bh_projs.append((bhx, bhy, einstein_r_px, bh_depth, shadow_r_px, bh))

        bh_projs.sort(key=lambda x: x[3])#nearest first

        #1 - Stars
        self._draw_stars(buf, camera, bh_projs)

        if show_particles:
            #2 - Far-side particles
            self._draw_particles(buf, camera, disk_list, free, 'far', bh_projs)
            #3 - Photon rings
            if Config.USE_VIRTUAL_ACCRETION_DISK:
                for bhx, bhy, _te, bh_depth, shadow_r_px, bh in bh_projs:
                    if shadow_r_px <= 0: continue
                    for disk in disk_list:
                        if disk.bh is bh and disk.n > 0:
                            self._draw_photon_ring(buf, camera, disk,
                                                   bhx, bhy, bh_depth, shadow_r_px)

        #4 - Commit numpy → surface
        np.clip(buf, 0, 255, out=buf)
        px = pygame.surfarray.pixels3d(surface);  px[:] = buf.astype(np.uint8);  del px

        #4b - Far-side celestial bodies
        for obj in objects:
            self._draw_celestial_body(surface, camera, obj, bh_projs, 'far')

        #5 - Merger flashes only (NO black circle — shadow is physically created
        #   by the photon ring + lensing occlusion, not a painted disc)
        for _bhx, _bhy, _te, _bd, _sr, bh in bh_projs:
            if bh.merge_flash > 0:
                self._draw_merger_flash(surface, bh, camera)

        #6 - Near-side particles
        if show_particles:
            px2  = pygame.surfarray.pixels3d(surface)
            buf2 = px2.astype(np.int16);  del px2
            self._draw_particles(buf2, camera, disk_list, free, 'near', bh_projs)
            np.clip(buf2, 0, 255, out=buf2)
            px3  = pygame.surfarray.pixels3d(surface);  px3[:] = buf2.astype(np.uint8);  del px3

        #6b - Near-side celestial bodies
        for obj in objects:
            self._draw_celestial_body(surface, camera, obj, bh_projs, 'near')

        #7 - Bloom
        self._bloom(surface)

        #8 - HUD + config panel
        self._hud(surface, fps, bh_list, show_particles, selected_bh_idx)
        if panel_state:
            self.draw_config_panel(surface, panel_state)
