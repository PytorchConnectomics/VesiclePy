# ves_vis

This module provides comprehensive visualization tools for vesicle and neuron data. It includes data conversion scripts, mesh generation utilities, and multiple visualization backends including HTML viewers, interactive Plotly plots, PyVista 3D scenes, and Neuroglancer integration.

## Quick Start

To immediately try the visualization pipeline with sample data:

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete pipeline (3 commands)
python scripts/conversion/dfGen.py sample/7-13_com_mapping.txt --output sample/sample_vesicles.parquet --types-dir sample/ --compute-neighbors
python scripts/conversion/neuron_mesh_gen.py sample/7-13_mask.h5 sample/sample_neuron.obj --format obj  
python scripts/visualizations/plotlyGen.py sample/sample_vesicles.parquet --neuron-mesh sample/sample_neuron.obj --sample-id 7-13 --color-by type --output sample/sample_visualization.html --swap-vesicle-xz

# Open sample/sample_visualization.html in your browser
```

## Setup

For setup, install the core dependencies and optional visualization packages as needed:
```bash
pip install -r requirements.txt
```

## Data Conversion

Convert raw vesicle mapping files into structured data formats for visualization.

### Extract Vesicle Data
Parse vesicle mapping files and generate parquet/CSV/JSON output:
```bash
python scripts/conversion/dfGen.py /path/to/mapping/files/ \
    --output vesicles.parquet \
    --types-dir /path/to/label/files/ \
    --compute-neighbors \
    --neighbor-radius 500
```

## Mesh Generation

Generate 3D meshes from volume data and vesicle coordinates.

### Neuron Meshes
Convert HDF5 neuron masks to 3D meshes using marching cubes:
```bash
python scripts/conversion/neuron_mesh_gen.py neuron_mask.h5 neuron.obj \
    --format glb \
    --apply-gaussian-filter \
    --fix-gaps-x-axis
```

### Vesicle Meshes
Generate sphere meshes from vesicle coordinate data:
```bash
python scripts/conversion/vesicle_mesh_gen.py vesicles.parquet \
    --output vesicles.obj \
    --format ply \
    --resolution 6 \
    --color-by type \
    --colormap plasma
```

## Visualization

Multiple visualization backends for different use cases and requirements.

### HTML Viewer
Create self-contained HTML viewer with Three.js:
```bash
python scripts/visualizations/htmlGen.py vesicles.parquet offsets.csv neurons/ colormap.json \
    --output my_viewer \
```

### Interactive Plotly
Generate interactive 3D plots for web browsers:
```bash
python scripts/visualizations/plotlyGen.py vesicles.parquet \
    --neuron-mesh neuron.obj \
    --sample-id SHL17 \
    --color-by type \
    --output interactive_plot.html \
    --swap-vesicle-xz
```

### PyVista 3D Rendering
Create high-quality 3D visualizations and screenshots:
```bash
python scripts/visualizations/pyvistaGen.py neurons/ vesicles/ offsets.csv \
    --no-interactive \
    --output scene.png \
    --camera-position xy
```

### Neuroglancer Viewer
Launch live Neuroglancer viewer for HDF5 volumes:
```bash
python scripts/visualizations/neuroglancerGen.py neuron_h5_files/ vesicles.h5 offsets.csv \
    --resolution 25 50 50 \
    --keep-open
```

## An Example Visualization On Sample Data

This complete pipeline demonstrates how to process raw vesicle mapping files into a final interactive 3D visualization using plotly. Follow these exact steps to reproduce the visualization:

For a large-scale visualization of the case study Hydra Vulgaris dataset, see: [https://akgohain.github.io/hydra_types/](https://akgohain.github.io/hydra_types/), where vesicles in all 20 neurons are colored by type.

### Prerequisites
```bash
# Set up Python environment and install dependencies
pip install -r requirements.txt
```

### Complete Pipeline

**Step 1: Extract vesicle data from mapping files**
```bash
python scripts/conversion/dfGen.py sample/7-13_com_mapping.txt \
    --output sample/sample_vesicles.parquet \
    --types-dir sample/ \
    --compute-neighbors
```
This processes the raw COM mapping file and type labels, computing vesicle coordinates, volumes, radii, types, and neighbor counts. The script automatically handles coordinate system conversion.

**Step 2: Generate neuron mesh from HDF5 mask**
```bash
python scripts/conversion/neuron_mesh_gen.py sample/7-13_mask.h5 sample/sample_neuron.obj \
    --format obj
```
This converts the 3D neuron mask volume into a surface mesh using marching cubes algorithm with preprocessing steps including binary closing, gap filling, and Gaussian filtering.

**Step 3: Create interactive visualization**
```bash
python scripts/visualizations/plotlyGen.py sample/sample_vesicles.parquet \
    --neuron-mesh sample/sample_neuron.obj \
    --sample-id 7-13 \
    --color-by type \
    --output sample/sample_visualization.html \
    --swap-vesicle-xz
```
This generates the final interactive 3D visualization with properly aligned vesicles and neuron mesh. The `--swap-vesicle-xz` flag is essential for correct spatial alignment.

### Results
- **sample_vesicles.parquet**: Processed vesicle data (204 vesicles, ~12KB)
- **sample_neuron.obj**: 3D neuron mesh (~82MB)
- **sample_visualization.html**: Interactive web-based 3D visualization (~67MB)

Open `sample_visualization.html` in your browser to view the interactive 3D visualization with:
- Vesicles colored by type using the viridis colormap
- Semi-transparent neuron mesh for spatial context
- Interactive controls for rotation, zoom, and inspection

## Troubleshooting

**Coordinate Alignment Issues**: If vesicles appear misaligned with the neuron mesh:
- Use `--swap-vesicle-xz` flag in plotlyGen.py (recommended for most cases)
- Use `--swap-mesh-xz` flag to swap mesh coordinates instead
- Use both flags if needed for specific coordinate systems

**Mesh Generation Warnings**: The "hole filling" warning is normal and doesn't affect the final visualization quality.

**Memory Issues**: For large datasets, the scripts automatically process data in chunks with garbage collection.

---

## API Reference

For detailed function documentation, see the individual script files:

- **Data Conversion**: [`dfGen.py`](scripts/conversion/dfGen.py) - Extract and process vesicle mapping data
- **Mesh Generation**: [`neuron_mesh_gen.py`](scripts/conversion/neuron_mesh_gen.py), [`vesicle_mesh_gen.py`](scripts/conversion/vesicle_mesh_gen.py)
- **Visualization**: [`htmlGen.py`](scripts/visualizations/htmlGen.py), [`plotlyGen.py`](scripts/visualizations/plotlyGen.py), [`pyvistaGen.py`](scripts/visualizations/pyvistaGen.py), [`neuroglancerGen.py`](scripts/visualizations/neuroglancerGen.py)

Each script includes comprehensive help accessible via `python script_name.py --help`.