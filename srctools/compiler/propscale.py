"""Implementation of CSGO's propcombine feature.

This merges static props together, so they can be drawn with a single
draw call.
"""
import operator
import os
import random
import colorsys
import shutil
import subprocess
import itertools
from collections import defaultdict
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import (
    Optional, Tuple, Callable, NamedTuple,
    FrozenSet, Dict, List, Set,
    Iterator, Union, MutableMapping,
)

from srctools import (
    Vec, VMF, Entity, conv_int, Angle, Matrix, FileSystemChain,
    Property, KeyValError,
)
from srctools.tokenizer import Tokenizer, Token
from srctools.game import Game

from srctools.logger import get_logger
from srctools.packlist import PackList
from srctools.bsp import BSP, StaticProp, StaticPropFlags, BModel, VisLeaf, VisTree
from srctools.mdl import Model, MDL_EXTS
from srctools.smd import Mesh
from srctools.compiler.mdl_compiler import ModelCompiler
from srctools.compiler.propcombine import (
    QC, unify_mdl, CollType, load_qcs, QC_TEMPLATE, QC_COLL_TEMPLATE,
    decompile_model,
)

LOGGER = get_logger(__name__)

SCALE_QC_TEMPLATE = '''\
$staticprop
$modelname "{path}"
$surfaceprop "{surf}"
    
$body body "reference.smd"

$contents {contents}

$bbox {bbox}

$sequence idle anim act_idle 1
'''

# Cache of the SMD models we have already parsed, so we don't need
# to parse them again. The second is the collision model.
_mesh_cache = {}  # type: Dict[Tuple[QC, int], Mesh]
_coll_cache = {}  # type: Dict[str, Mesh]


class ScalePropPos(NamedTuple):
    model: str
    skin: int
    scale: float
    solidity: CollType


def scale_prop(
    compiler: ModelCompiler,
    prop: StaticProp,
    vistrees: list[VisTree],
    lookup_model: Callable[[str], Tuple[QC, Model]],
) -> StaticProp:
    """Scale the given prop, compiling a model if required."""

    try:
        coll = CollType(prop.solidity)
    except ValueError:
        raise ValueError(
             'Unknown prop_static collision type '
             '{} for "{}" at {}!'.format(
                prop.solidity,
                prop.model,
                prop.origin,
             )
        )

    prop_pos = ScalePropPos(
        prop.model,
        prop.skin,
        prop.scaling,
        coll,
    )

    # We don't want to build collisions if it's not used.
    has_coll = prop_pos.solidity is not CollType.NONE
    mdl_name, result = compiler.get_model(
        (prop_pos, has_coll),
        compile_func, lookup_model,
    )

    _, mdl = lookup_model(mdl_name)

    visleafs = compute_static_prop_leaves(mdl, prop.origin, prop.angles, vistrees)

    # Many of these we require to be the same, so we can read them
    # from any of the component props.
    return StaticProp(
        model=mdl_name,
        origin=prop.origin,
        angles=prop.angles,
        scaling=1.0,
        visleafs=visleafs,
        solidity=(CollType.VPHYS if has_coll else CollType.NONE).value,
        flags=prop.flags,
        lighting=prop.lighting,
        tint=prop.tint,
        renderfx=prop.renderfx,
    )


def compile_func(
    mdl_key: Tuple[ScalePropPos, bool],
    temp_folder: Path,
    mdl_name: str,
    lookup_model: Callable[[str], Tuple[QC, Model]],
) -> None:
    """Build this scaled model."""
    LOGGER.info('Compiling {}...', mdl_name)
    prop_pos, has_coll = mdl_key

    # Unify these properties.
    cdmats = set()  # type: Set[str]

    qc, mdl = lookup_model(prop_pos.model)
    assert mdl is not None, prop_pos.model
    surfprop = mdl.surfaceprop.casefold()
    cdmats.update(mdl.cdmaterials)
    phy_content_type = mdl.contents

    ref_mesh = Mesh.blank('static_prop')
    coll_mesh = None  # type: Optional[Mesh]

    qc, mdl = lookup_model(prop_pos.model)
    try:
        child_ref = _mesh_cache[qc, prop_pos.skin]
    except KeyError:
        LOGGER.info('Parsing ref "{}"', qc.ref_smd)
        with open(qc.ref_smd, 'rb') as fb:
            child_ref = Mesh.parse_smd(fb)

        if prop_pos.skin != 0 and prop_pos.skin < len(mdl.skins):
            # We need to rename the materials to match the skin.
            swap_skins = dict(zip(
                mdl.skins[0],
                mdl.skins[prop_pos.skin]
            ))
            for tri in child_ref.triangles:
                tri.mat = swap_skins.get(tri.mat, tri.mat)

        _mesh_cache[qc, prop_pos.skin] = child_ref

    child_coll = build_collision(qc, prop_pos, child_ref)

    offset = Vec(0, 0, 0)
    angles = Angle(0, 0, 0)

    ref_mesh.append_model(child_ref, angles, offset, prop_pos.scale * qc.ref_scale)

    if has_coll and child_coll is not None:
        if coll_mesh is None:
            coll_mesh = Mesh.blank('static_prop')
        coll_mesh.append_model(child_coll, angles, offset, prop_pos.scale * qc.phy_scale)

    with (temp_folder / 'reference.smd').open('wb') as fb:
        ref_mesh.export(fb)

    # Generate  a  blank animation.
    with (temp_folder / 'anim.smd').open('wb') as fb:
        Mesh.blank('static_prop').export(fb)

    if coll_mesh is not None:
        with (temp_folder / 'physics.smd').open('wb') as fb:
            coll_mesh.export(fb)

    bbox_min, bbox_max = Vec.bbox(
        vert.pos
        for tri in
        ref_mesh.triangles
        for vert in tri
    )

    with (temp_folder / 'model.qc').open('w') as f:
        f.write(SCALE_QC_TEMPLATE.format(
            path=mdl_name,
            surf=surfprop,
            # For $contents, we need to decompose out each bit.
            # This is the same as BSP's flags in public/bsp_flags.h
            # However only a few types are allowable.
            contents=' '.join([
                cont
                for mask, cont in [
                    (0x1, '"solid"'),
                    (0x8, '"grate"'),
                    (0x2000000, '"monster"'),
                    (0x20000000, '"ladder"'),
                ]
                if mask & phy_content_type
                # 0 needs to produce this value.
            ]) or '"notsolid"',

            bbox=' '.join([
                str(bbox_min * prop_pos.scale),
                str(bbox_max * prop_pos.scale),
            ])
        ))

        for mat in sorted(cdmats):
            f.write('$cdmaterials "{}"\n'.format(mat))

        if coll_mesh is not None:
            f.write(QC_COLL_TEMPLATE)


def build_collision(qc: QC, prop: ScalePropPos, ref_mesh: Mesh) -> Optional[Mesh]:
    """Get the correct collision mesh for this model."""
    if prop.solidity is CollType.NONE:  # Non-solid
        return None
    elif prop.solidity is CollType.VPHYS or prop.solidity is CollType.BSP:
        if qc.phy_smd is None:
            return None
        try:
            return _coll_cache[qc.phy_smd]
        except KeyError:
            LOGGER.info('Parsing coll "{}"', qc.phy_smd)
            with open(qc.phy_smd, 'rb') as fb:
                coll = Mesh.parse_smd(fb)

            _coll_cache[qc.phy_smd] = coll
            return coll
    # Else, it's one of the three bounding box types.
    # We don't really care about which.
    bbox_min, bbox_max = Vec.bbox(
        vert.pos
        for tri in
        ref_mesh.triangles
        for vert in tri
    )

    bbox_min *= prop.scale
    bbox_max *= prop.scale

    return Mesh.build_bbox('static_prop', 'phy', bbox_min, bbox_max)


def compute_static_prop_leaves(
    model: Model,
    origin: Vec,
    angles: Angle,
    vistrees: List[VisTree]
) -> Set[VisLeaf]:

    rot_min: Vec = model.hull_min @ angles + origin
    rot_max: Vec = model.hull_max @ angles + origin

    bbox_min, bbox_max = Vec.bbox(rot_min, rot_max)

    def get_all_bbox_corners(vec_min: Vec, vec_max: Vec) -> Iterator[Vec]:
        for i in range(0, 8):
            out_vec = Vec(0, 0, 0)

            if i & 0x100 > 0:
                out_vec.x = vec_min.x
            else:
                out_vec.x = vec_max.x

            if i & 0x010 > 0:
                out_vec.y = vec_min.y
            else:
                out_vec.y = vec_max.y

            if i & 0x001 > 0:
                out_vec.z = vec_min.z
            else:
                out_vec.z = vec_max.z

            yield out_vec

    prop_leaves: set[VisLeaf] = set()

    for tree in vistrees:

        for point in get_all_bbox_corners(bbox_min, bbox_max):
            point_test = tree.test_point(point)
            if point_test is not None:
                prop_leaves.add(point_test)

        origin_test = tree.test_point(origin)
        if origin_test is not None:
            prop_leaves.add(origin_test)

    return set(prop_leaves)


def scale_props(
        bsp: BSP,
        bsp_ents: VMF,
        pack: PackList,
        game: Game,
        studiomdl_loc: Path,
        *,
        qc_folders: List[Path] = None,
        crowbar_loc: Optional[Path] = None,
        decomp_cache_loc: Path = None,
        debug_tint: bool = False,
        debug_dump: bool = False,
) -> None:
    """Scale the static scalable props in this map."""
    possible_scalable_props = list(bsp_ents.by_class['prop_static_scalable'])
    scalable_props: list[Entity] = []
    for prop in possible_scalable_props:
        scale: int
        try:
            scale = int(float(prop['modelscale']))
        except KeyError:
            continue

        scalable_props.append(prop)

    LOGGER.info('{} scalable props found.', len(scalable_props))

    if not qc_folders and decomp_cache_loc is None:
        # If gameinfo is blah/game/hl2/gameinfo.txt,
        # QCs should be in blah/content/ according to Valve's scheme.
        # But allow users to override this.
        # If Crowbar's path is provided, that means they may want to just supply nothing.
        qc_folders = [game.path.parent.parent / 'content']

    # Parse through all the QC files.
    LOGGER.info('Parsing QC files. Paths: \n{}', '\n'.join(map(str, qc_folders)))
    qc_map: Dict[str, Optional[QC]] = {}
    for qc_folder in qc_folders:
        load_qcs(qc_map, qc_folder)
    LOGGER.info('Done! {} props.', len(qc_map))

    # Don't re-parse models continually.
    mdl_map: Dict[str, Optional[Model]] = {}

    def get_model_only(filename: str) -> Union[Model, None]:
        """Given a filename, load/parse the MDL data.
        """
        key = unify_mdl(filename)
        try:
            model = mdl_map[key]
        except KeyError:
            try:
                mdl_file = pack.fsys[filename]
            except FileNotFoundError:
                # We don't have this model, we can't scale...
                return None
            model = mdl_map[key] = Model(pack.fsys, mdl_file)
            if 'no_scale' in model.keyvalues.casefold():
                mdl_map[key] = qc_map[key] = None
                return None
        if model is None:
            return None

        return model

    scaled_props: list[StaticProp] = []

    if not studiomdl_loc.exists():
        LOGGER.warning('No studioMDL! Cannot scale props! Using a scale of 1.0 instead.')

        for ent in scalable_props:
            modelname = ent['model']
            origin = Vec.from_str(ent['origin'])
            angles = Angle.from_str(ent['angles'])

            prop_model = get_model_only(modelname)
            if prop_model is None:
                continue

            visleafs = compute_static_prop_leaves(prop_model, origin, angles, bsp.nodes)

            r, g, b = 1.0, 1.0, 1.0

            scaled_props.append(StaticProp(
                model=modelname,
                origin=origin,
                angles=angles,
                scaling=1.0,
                visleafs=visleafs,
                solidity=int(float(ent['solid'])),
                flags=StaticPropFlags(0x100),
                lighting=origin,
                tint=Vec(round(r * 255), round(g * 255), round(b * 255)),
                renderfx=255,
            ))

    map_name = Path(bsp.filename).stem

    # Wipe these, if they're being used again.
    _mesh_cache.clear()
    _coll_cache.clear()
    missing_qcs: Set[str] = set()

    # This stores all of the props in the map
    final_props: list[StaticProp] = bsp.props

    def get_model(filename: str) -> Union[Tuple[QC, Model], Tuple[None, None]]:
        """Given a filename, load/parse the QC and MDL data.

        Either both are returned, or neither are.
        """
        key = unify_mdl(filename)
        try:
            model = mdl_map[key]
        except KeyError:
            try:
                mdl_file = pack.fsys[filename]
            except FileNotFoundError:
                # We don't have this model, we can't combine...
                return None, None
            model = mdl_map[key] = Model(pack.fsys, mdl_file)
            if 'no_scale' in model.keyvalues.casefold():
                mdl_map[key] = qc_map[key] = None
                return None, None
        if model is None or key in missing_qcs:
            return None, None

        try:
            qc = qc_map[key]
        except KeyError:
            if crowbar_loc is None:
                missing_qcs.add(key)
                return None, None
            qc = decompile_model(pack.fsys, decomp_cache_loc, crowbar_loc, filename, model.checksum)
            qc_map[key] = qc

        if qc is None:
            return None, None
        else:
            return qc, model

    # Ignore these two, they don't affect our new prop.
    relevant_flags = ~(StaticPropFlags.HAS_LIGHTING_ORIGIN | StaticPropFlags.DOES_FADE)

    scale_count = 0
    with ModelCompiler(
            game,
            studiomdl_loc,
            pack,
            map_name,
            'propcombine',
    ) as compiler:
        for scalable_prop in scalable_props:

            modelname = scalable_prop['model']
            origin = Vec.from_str(scalable_prop['origin'])
            angles = Angle.from_str(scalable_prop['angles'])

            prop_model = get_model_only(modelname)
            if prop_model is None:
                continue

            r, g, b = 1.0, 1.0, 1.0

            prop_to_scale = StaticProp(
                model=modelname,
                origin=origin,
                angles=angles,
                scaling=float(scalable_prop['modelscale']),
                visleafs=set(),
                solidity=int(float(scalable_prop['solid'])),
                flags=StaticPropFlags(0x100),
                lighting=origin,
                tint=Vec(round(r * 255), round(g * 255), round(b * 255)),
                renderfx=255,
            )

            scaled_prop = scale_prop(compiler, prop_to_scale, bsp.nodes, get_model)
            if debug_tint:
                # Compute a random hue, and convert back to RGB 0-255.
                r, g, b = colorsys.hsv_to_rgb(random.random(), 1, 1)
                scaled_prop.tint = Vec(round(r * 255), round(g * 255), round(b * 255))
            final_props.append(scaled_prop)
            scale_count += 1

    LOGGER.info(
        'Scaled {} props - {} failed',
        scale_count,
        len(scalable_props) - scale_count,
    )

    LOGGER.debug('Models with unknown QCs: \n{}', '\n'.join(sorted(missing_qcs)))
    # If present, delete old cache file. We'll have cleaned up the models.
    try:
        os.remove(compiler.model_folder_abs / 'cache.vdf')
    except FileNotFoundError:
        pass

    # Clean up the map from scalable props
    LOGGER.info('Removing {} prop_static_scalables....', len(possible_scalable_props))
    for prop in possible_scalable_props:
        prop.remove()

    bsp.props = final_props
