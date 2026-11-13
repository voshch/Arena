import math
import pathlib
from collections.abc import Sequence

import attrs
import numpy as np
import numpy.typing as npt

from arena_fleet.config import CostConfig, FlowConfig
from arena_fleet.geometry import Wall

Person = tuple[float, float, float, float, float]


@attrs.frozen
class MapOfDynamics:
    """Learned pedestrian density per time window and cell (ped/m^2) of robot-free runs."""

    rho: npt.NDArray[np.float64]
    dt: float
    cell: float

    @classmethod
    def load(cls, path: pathlib.Path) -> "MapOfDynamics":
        z = np.load(path, allow_pickle=False)
        x_edges, t_edges = z["x_edges"], z["t_edges"]
        return cls(rho=np.asarray(z["rho"], dtype=np.float64), dt=float(t_edges[1] - t_edges[0]), cell=float(x_edges[1] - x_edges[0]))


def disk(r: float, cell: float) -> list[tuple[int, int]]:
    k = int(math.ceil(r / cell))
    return [(i, j) for i in range(-k, k + 1) for j in range(-k, k + 1) if math.hypot(i, j) * cell <= r]


class PeopleView:
    """Occupancy layers k * dt s ahead; every information level only changes how the layers are made."""

    def __init__(self, cost: CostConfig, walls: Sequence[Wall], mod: MapOfDynamics | None = None) -> None:
        v = cost["people_view"]
        self.cell, self.dt = v["grid"]["cell_m"], v["dt_s"]
        self.nx, self.ny = int(round(v["grid"]["x_max_m"] / self.cell)), int(round(v["grid"]["y_max_m"] / self.cell))
        self.n = int(cost["encounter"]["horizon_s"] / self.dt) + 1
        self.walls, self.mod = list(walls), mod
        self.flow: FlowConfig = cost["flow"]
        self.person_r = v["person_r_m"]
        self.person, self.robot = disk(self.person_r, self.cell), disk(v["robot_r_m"], self.cell)
        self.layers: list[set[int]] = [set() for _ in range(self.n)]
        self.heat: dict[int, list[list[int]]] = {}
        self.probs: list[npt.NDArray[np.float32]] | None = None

    def idx(self, x: float, y: float) -> int | None:
        i, j = int(x // self.cell), int(y // self.cell)
        return j * self.nx + i if 0 <= i < self.nx and 0 <= j < self.ny else None

    def _k(self, tau: float) -> int:
        return max(0, min(int(round(tau / self.dt)), self.n - 1))

    def layer(self, tau: float) -> set[int]:
        return self.layers[self._k(tau)]

    def occupied(self, tau: float, x: float, y: float, inflate: bool = True) -> bool:
        """Is (x, y) occupied tau s ahead; inflate tests a robot body instead of a point."""
        lay = self.layer(tau)
        if not inflate:
            return self.idx(x, y) in lay
        i0, j0 = int(x // self.cell), int(y // self.cell)
        return any((j0 + j) * self.nx + i0 + i in lay for i, j in self.robot if 0 <= i0 + i < self.nx and 0 <= j0 + j < self.ny)

    def count(self, tau: float, cx: float, cy: float, r: float) -> float:
        """People-equivalents in a circle: occupied area over one person's disk."""
        lay = self.layer(tau)
        k, c, hits = int(r / self.cell) + 1, self.cell, 0
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x, y = cx + i * c, cy + j * c
                if math.hypot(x - cx, y - cy) <= r and self.idx(x, y) in lay:
                    hits += 1
        return hits * c * c / (math.pi * self.person_r**2)

    def prob(self, tau: float, x: float, y: float) -> float:
        i = self.idx(x, y)
        if i is None:
            return 0.0
        if self.probs is None:
            return 1.0 if i in self.layer(tau) else 0.0
        return float(self.probs[self._k(tau)][i])

    def t_wall(self, x: float, y: float, vx: float, vy: float) -> float:
        """Seconds until a person walking straight reaches a wall."""
        best = math.inf
        for (ax, ay), (bx, by) in self.walls:
            sx, sy = bx - ax, by - ay
            den = vx * sy - vy * sx
            if abs(den) < 1e-9:
                continue
            t = ((ax - x) * sy - (ay - y) * sx) / den
            u = ((ax - x) * vy - (ay - y) * vx) / den
            if 0.0 <= t < best and 0.0 <= u <= 1.0:
                best = t
        return best

    def update_people(self, people: Sequence[Person]) -> None:
        """Constant-velocity forecast of known people (x, y, vx, vy, age_s), stopped at walls."""
        self.layers = [set() for _ in range(self.n)]
        for x, y, vx, vy, age in people:
            tw = self.t_wall(x, y, vx, vy)
            for k in range(self.n):
                t = min(k * self.dt + age, max(tw - 0.05, 0.0))
                i0, j0 = int((x + vx * t) // self.cell), int((y + vy * t) // self.cell)
                lay = self.layers[k]
                for i, j in self.person:
                    if 0 <= i0 + i < self.nx and 0 <= j0 + j < self.ny:
                        lay.add((j0 + j) * self.nx + i0 + i)
        self.heat = {k: [[c, 100] for c in sorted(self.layers[k])] for k in (0, 5) if k < self.n}

    def update_flow(self, people: Sequence[Person], t_rel: float, live: bool = True) -> None:
        """Live Gaussians blended into the learned map, thresholded to occupancy; live=False uses the map alone."""
        mod = self.mod
        if mod is None:
            raise ValueError("the flow and mod information levels need a map of dynamics")
        f, c = self.flow, self.cell
        area = math.pi * f["footprint_r_m"] ** 2
        heat: dict[int, list[list[int]]] = {}
        layers: list[set[int]] = []
        probs: list[npt.NDArray[np.float32]] = []
        for k in range(self.n):
            tau = k * self.dt
            w = math.exp(-tau / f["blend_tau_s"]) if live else 0.0
            win = min(max(int((t_rel + tau) // mod.dt), 0), mod.rho.shape[0] - 1)
            rho = (1.0 - w) * np.asarray(mod.rho[win][: self.ny, : self.nx], dtype=np.float64)
            if w >= 0.01:
                sig = f["sigma0_m"] + f["sigma_per_s"] * tau
                rad = int(math.ceil(3 * sig / c))
                for x, y, vx, vy, age in people:
                    t = min(tau + age, max(self.t_wall(x, y, vx, vy) - 0.05, 0.0))
                    px, py = x + vx * t, y + vy * t
                    i0, j0 = int(px // c), int(py // c)
                    a0, a1, b0, b1 = max(i0 - rad, 0), min(i0 + rad + 1, self.nx), max(j0 - rad, 0), min(j0 + rad + 1, self.ny)
                    if a0 >= a1 or b0 >= b1:
                        continue
                    xs = (np.arange(a0, a1) + 0.5) * c - px
                    ys = (np.arange(b0, b1) + 0.5) * c - py
                    g = np.exp(-(xs**2)[None, :] / (2 * sig * sig) - (ys**2)[:, None] / (2 * sig * sig)) / (2 * math.pi * sig * sig)
                    rho[b0:b1, a0:a1] += w * g
            p = (1.0 - np.exp(-rho * area)).ravel()
            probs.append(p.astype(np.float32))
            layers.append({int(i) for i in np.flatnonzero(p >= f["threshold"])})
            if k in (0, 5):
                heat[k] = [[int(i), int(p[i] * 100)] for i in np.flatnonzero(p >= 0.05)]
        self.layers, self.heat, self.probs = layers, heat, probs
