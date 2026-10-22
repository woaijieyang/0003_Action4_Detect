"""Conservative cached collision gate for complete, rigid template actors.

No anatomy is closed or invented. Open nonplanar meshes have unknown volume:
overlapping bounds are NOT promoted by zero surface hits. Closed shells use
component-wise ray parity. This is an engineering gate, not mechanical approval.
"""
import hashlib
import time
from dataclasses import dataclass
import numpy as np

GEOMETRY_TYPES = {'MESH', 'CURVE', 'SURFACE', 'FONT', 'META'}
TOL = 2e-7  # metre-scale numerical ambiguity is rejected, never cleared


def transform(matrix, vertices):
    return np.asarray(vertices) @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


def components_and_closed(vertices, faces):
    """Topology on evaluated triangles; one representative per connected surface."""
    n = len(vertices)
    parent = np.arange(n)
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for a, b, c in faces:
        ra = root(a)
        parent[root(b)] = ra
        parent[root(c)] = ra
    reps = {}
    for i in np.unique(faces):
        reps.setdefault(root(i), int(i))
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, inverse, count = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    orientation = np.bincount(inverse, weights=np.where(edges[:, 0] < edges[:, 1], 1, -1))
    closed = bool(np.all(count == 2) and np.all(orientation == 0))
    return list(reps.values()), closed, int(np.sum(count == 1))


def triangle_contact(a, b):
    """Conservative full triangle SAT, including coplanar in-plane axes."""
    ea = np.roll(a, -1, axis=0) - a
    eb = np.roll(b, -1, axis=0) - b
    na, nb = np.cross(ea[0], ea[1]), np.cross(eb[0], eb[1])
    axes = [na, nb]
    axes.extend(np.cross(x, y) for x in ea for y in eb)
    axes.extend(np.cross(na, x) for x in ea)
    axes.extend(np.cross(nb, x) for x in eb)
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm < 1e-15:
            continue
        axis = axis / norm
        pa, pb = a @ axis, b @ axis
        if max(pb.min() - pa.max(), pa.min() - pb.max()) > TOL:
            return False
    return True


def point_in_closed(point, vertices, faces):
    """Ray parity with three directions; edge/near-surface ambiguity fails closed."""
    triangles = vertices[faces]
    v0 = triangles[:, 0]
    e1, e2 = triangles[:, 1] - v0, triangles[:, 2] - v0
    answers = []
    for direction in ((1., .371, .529), (.257, 1., .613), (.443, .193, 1.)):
        d = np.array(direction)
        d /= np.linalg.norm(d)
        h = np.cross(d, e2)
        det = np.einsum('ij,ij->i', e1, h)
        active = np.abs(det) > 1e-10 * np.linalg.norm(np.cross(e1, e2), axis=1)
        inv = np.zeros_like(det)
        inv[active] = 1. / det[active]
        s = point - v0
        u = inv * np.einsum('ij,ij->i', s, h)
        q = np.cross(s, e1)
        v = inv * (q @ d)
        distance = inv * np.einsum('ij,ij->i', e2, q)
        hits = active & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (distance >= -TOL)
        if np.any(hits & ((distance <= TOL) | (u < 1e-8) | (v < 1e-8) | (1-u-v < 1e-8))):
            continue
        answers.append(bool(np.count_nonzero(hits) % 2))
    if len(answers) < 2 or len(set(answers)) != 1:
        return None
    return answers[0]


@dataclass
class Geometry:
    vertices: np.ndarray
    faces: np.ndarray
    representatives: list
    closed: bool
    planar: bool
    boundary_edges: int
    tree: object = None

    @classmethod
    def create(cls, vertices, faces):
        vertices, faces = np.asarray(vertices, dtype=float), np.asarray(faces, dtype=int)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
            raise ValueError('Finite nonempty evaluated vertices required')
        if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
            raise ValueError('Valid evaluated triangles required')
        area = np.linalg.norm(np.cross(vertices[faces[:, 1]]-vertices[faces[:, 0]], vertices[faces[:, 2]]-vertices[faces[:, 0]]), axis=1)
        if np.any(area < 1e-15):
            raise ValueError('Degenerate evaluated triangle: cannot certify')
        reps, closed, boundary = components_and_closed(vertices, faces)
        singular = np.linalg.svd(vertices - vertices.mean(0), compute_uv=False)
        planar = bool(not closed and singular[-1] <= TOL * np.sqrt(len(vertices)))
        return cls(vertices, faces, reps, closed, planar, boundary)

    def moved(self, matrix):
        return Geometry(transform(matrix, self.vertices), self.faces, self.representatives,
                        self.closed, self.planar, self.boundary_edges)

    def bvh(self):
        if self.tree is None:
            from mathutils.bvhtree import BVHTree
            self.tree = BVHTree.FromPolygons(self.vertices.tolist(), self.faces.tolist(), all_triangles=True, epsilon=0.)
        return self.tree


def separated_axes(a, b, axes):
    """Sufficient separation proof only; overlap never means collision."""
    for axis in axes:
        length = np.linalg.norm(axis)
        if length < 1e-12:
            continue
        axis = axis / length
        pa, pb = a @ axis, b @ axis
        gap = max(pb.min()-pa.max(), pa.min()-pb.max())
        if gap > TOL:
            return {'axis': axis.tolist(), 'gap_m': float(gap)}
    return None


def check_pair(a, b, axes=()):
    from data.action4_modular_bilateral_scene import aabb_separation, filter_separated_bvh_candidates
    from data.action4_mount_modules import classify_pairs
    separation = aabb_separation(a.vertices, b.vertices)
    # A tiny gap stays ambiguous under the declared engineering tolerance.
    if max(separation['axis_gaps_m']) > TOL:
        return {'status': 'CLEAR_AABB', 'separation': separation}
    proof = separated_axes(a.vertices, b.vertices, axes)
    if proof:
        return {'status': 'CLEAR_SEPARATING_AXIS', 'separation': proof}
    candidates = a.bvh().overlap(b.bvh())
    retained, removed = filter_separated_bvh_candidates(a.vertices, a.faces, b.vertices, b.faces, candidates)
    actual = [(i, j) for i, j in retained if triangle_contact(a.vertices[a.faces[i]], b.vertices[b.faces[j]])]
    if actual:
        classification = classify_pairs(a.vertices, a.faces, b.vertices, b.faces, actual)
        return {'status': 'PROHIBITED_SURFACE_CONTACT', 'triangle_pairs_count': len(actual),
                'triangle_pairs_first16': classification['outside_or_unresolved_pairs'][:16],
                'bvh_candidates': len(candidates), 'separated_candidates': len(removed)}
    directions = []
    for source, container, direction in ((a, b, 'a_in_b'), (b, a, 'b_in_a')):
        if container.planar:
            directions.append({'direction': direction, 'status': 'ZERO_THICKNESS_SHEET'})
            continue
        if not container.closed:
            directions.append({'direction': direction, 'status': 'UNKNOWN_OPEN_VOLUME'})
            continue
        answers = [point_in_closed(source.vertices[i], container.vertices, container.faces) for i in source.representatives]
        if any(x is True for x in answers):
            return {'status': 'PROHIBITED_CONTAINMENT', 'direction': direction,
                    'components_checked': len(answers), 'inside_components': [i for i, x in enumerate(answers) if x is True]}
        directions.append({'direction': direction,
                           'status': 'UNKNOWN_RAY_PARITY' if any(x is None for x in answers) else 'OUTSIDE_CLOSED_SOLID',
                           'components_checked': len(answers)})
    unknown = any(row['status'].startswith('UNKNOWN') for row in directions)
    return {'status': 'UNKNOWN_CONTAINMENT' if unknown else 'CLEAR_SURFACE_AND_CONTAINMENT',
            'surface_triangle_pairs_count': 0, 'containment': directions,
            'bvh_candidates': len(candidates), 'separated_candidates': len(removed)}


def mesh_signature(obj):
    """Fast source-buffer fingerprint; no material graph/hash, no file rereading."""
    if obj.type != 'MESH' or len(obj.modifiers) or obj.data.shape_keys is not None:
        return None
    mesh = obj.data
    vertices = np.empty(len(mesh.vertices)*3, dtype=np.float32)
    loops = np.empty(len(mesh.loops), dtype=np.int32)
    sizes = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.vertices.foreach_get('co', vertices)
    mesh.loops.foreach_get('vertex_index', loops)
    mesh.polygons.foreach_get('loop_total', sizes)
    return hashlib.sha256(vertices.tobytes()+loops.tobytes()+sizes.tobytes()).hexdigest()


def verify_evaluation_supported(obj):
    if obj.type not in GEOMETRY_TYPES | {'EMPTY', 'CAMERA', 'LIGHT'}:
        raise ValueError('Unsupported renderable object type: '+obj.name)
    if obj.is_instancer:
        raise ValueError('Instanced geometry requires explicit expansion: '+obj.name)
    for modifier in obj.modifiers:
        if modifier.show_render != modifier.show_viewport:
            raise ValueError('Render/viewport modifier visibility differs: '+obj.name)
        if hasattr(modifier, 'render_levels') and modifier.render_levels != modifier.levels:
            raise ValueError('Render/viewport subdivision differs: '+obj.name)
    if obj.type in {'CURVE', 'SURFACE'}:
        if obj.data.render_resolution_u not in (0, obj.data.resolution_u):
            raise ValueError('Render/viewport curve resolution differs: '+obj.name)


class TemplateCollisionChecker:
    """Construct once immediately after TemplateRuntime, before first apply.

    Static-static and same-actor pairs reuse the supplied baseline, conditional
    on exact cached geometry and rigid relative transforms. All changed pairs
    are checked; report.accepted only means this incremental gate cleared.
    """
    def __init__(self, runtime):
        from data.action4_modular_assembly import world_geometry
        if runtime.current is not None:
            raise ValueError('Initialize collision cache at unchanged template baseline')
        self.runtime = runtime
        self.cache, self.signatures, self.groups = {}, {}, {}
        self.group_names = {name: side for side, g in runtime.metadata['actor_groups'].items() for name in g['objects']}
        if sum(len(g['objects']) for g in runtime.metadata['actor_groups'].values()) != len(self.group_names):
            raise ValueError('Overlapping actor membership')
        self.excluded_baseline_hidden = []
        for obj in runtime.scene.objects:
            verify_evaluation_supported(obj)
            if obj.name in {p.name for p in runtime.patches.values()} or obj.type not in GEOMETRY_TYPES:
                continue
            if obj.hide_render:
                self.excluded_baseline_hidden.append(obj.name)
                continue
            vertices, faces = world_geometry(obj)
            self.cache[obj.name] = Geometry.create(vertices, faces)
            self.signatures[obj.name] = mesh_signature(obj)
            self.groups[obj.name] = self.group_names.get(obj.name, 'static')
        self.topology = {n: {'triangles': len(g.faces), 'components': len(g.representatives),
                             'closed': g.closed, 'planar': g.planar, 'boundary_edges': g.boundary_edges}
                         for n, g in self.cache.items()}

    def check(self):
        started = time.monotonic()
        report = {'schema': 'action4.template_collision.v1', 'accepted': False,
                  'collision_checked': True, 'training_accepted': False,
                  'baseline_internal_reuse': 'Rigid invariance only; no new mechanical or anatomy acceptance',
                  'prohibited': [], 'unknown': [], 'pair_status_counts': {},
                  'tolerance_m': TOL}
        try:
            self._check(report)
        except (ValueError, RuntimeError, KeyError) as exc:
            report['unknown'].append({'status': 'UNKNOWN_GEOMETRY_OR_INVARIANT', 'reason': str(exc)})
        report['accepted'] = not report['prohibited'] and not report['unknown']
        report['status'] = 'PASS_INCREMENTAL_COLLISION' if report['accepted'] else 'REJECT_COLLISION' if report['prohibited'] else 'REJECT_UNRESOLVED'
        report['elapsed_seconds'] = time.monotonic() - started
        return report

    def _check(self, report):
        from data.action4_modular_assembly import world_geometry
        r = self.runtime
        r.verify(r.scene)
        geometry, matrices = {}, {}
        actor_delta = {}
        for name in self.excluded_baseline_hidden:
            if not r.scene.objects[name].hide_render:
                raise ValueError('Baseline-excluded geometry became visible: '+name)
        for name, base in self.cache.items():
            obj = r.scene.objects[name]
            verify_evaluation_supported(obj)
            signature = mesh_signature(obj)
            if signature != self.signatures[name]:
                raise ValueError('Cached source geometry changed: '+name)
            delta = np.asarray(obj.matrix_world) @ np.linalg.inv(r.world[name])
            if not np.allclose(delta[:3, :3].T@delta[:3, :3], np.eye(3), atol=2e-5, rtol=0) or not np.isclose(np.linalg.det(delta[:3,:3]), 1., atol=2e-5):
                raise ValueError('Nonrigid transform: '+name)
            if signature is None:
                v, f = world_geometry(obj)
                if not np.array_equal(f, base.faces) or not np.allclose(v, transform(delta, base.vertices), atol=3e-6, rtol=0):
                    raise ValueError('Evaluated curve/modifier geometry changed: '+name)
            group = self.groups[name]
            if group == 'static' and not np.allclose(delta, np.eye(4), atol=3e-6, rtol=0):
                raise ValueError('Static cache moved after initialization: '+name)
            if group in actor_delta and not np.allclose(actor_delta[group], delta, atol=5e-6, rtol=0):
                raise ValueError('Relative actor/static transform changed: '+name)
            actor_delta[group] = delta
            if obj.hide_render:
                continue
            geometry[name] = base if group == 'static' else base.moved(delta)
            matrices[name] = delta[:3, :3]
        groups = dict(self.groups)
        for side, patch in r.patches.items():
            if patch.hide_render:
                continue
            verify_evaluation_supported(patch)
            if patch.type != 'MESH':
                raise ValueError('Occluder must remain a mesh sheet: '+patch.name)
            vertices, faces = world_geometry(patch)
            geometry[patch.name] = Geometry.create(vertices, faces)
            if not geometry[patch.name].planar:
                raise ValueError('Occluder is not a planar sheet')
            groups[patch.name] = patch.name
            matrices[patch.name] = np.eye(3)
        names = sorted(geometry)
        reused = checked = 0
        for i, a in enumerate(names):
            for b in names[i+1:]:
                if groups[a] == groups[b]:
                    reused += 1
                    continue
                checked += 1
                axes = list(matrices[a].T) + list(matrices[b].T)
                axes += [np.cross(x, y) for x in matrices[a].T for y in matrices[b].T]
                result = check_pair(geometry[a], geometry[b], axes)
                status = result['status']
                report['pair_status_counts'][status] = report['pair_status_counts'].get(status, 0) + 1
                if status.startswith('PROHIBITED'):
                    report['prohibited'].append({'objects': [a, b], **result})
                elif status.startswith('UNKNOWN'):
                    report['unknown'].append({'objects': [a, b], **result})
        report.update(checked_changed_pairs=checked, baseline_invariant_pairs_reused=reused,
                      active_geometry_objects=len(geometry), finite_geometry=True)

