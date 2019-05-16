#make it easy to access submodules
import sys, os
sys.path.insert(1, os.path.join(os.getcwd(), 'nv_wavenet', 'pytorch'))
sys.path.insert(1, os.path.join(os.getcwd(), 'Ensemble'))
