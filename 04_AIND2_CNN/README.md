# aind2-cnn

### Instructions

1. Clone the repository and navigate to the downloaded folder.
	
```	
git clone https://github.com/udacity/aind2-cnn.git
cd aind2-cnn
```

2. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

3. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```
	
4. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```  
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
5. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
6. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

7. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

8. Launch Jupyter notebook.
```
jupyter notebook
```

9. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). 

</div>
<div class="divider"></div><div class="ud-atom">
  <h3></h3>
  <div>
  <h1 id="dimensionality">Dimensionality</h1>
<p>Just as with neural networks, we create a CNN in Keras by first creating a <code>Sequential</code> model.</p>
<p>We add layers to the network by using the <code>.add()</code> method.</p>
<p>Copy and paste the following code into a Python executable named <code>conv-dims.py</code>:</p>
<pre><code class="python language-python">from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, strides=2, padding='valid', 
    activation='relu', input_shape=(200, 200, 1)))
model.summary()</code></pre>
<p>We will not train this CNN; instead, we'll use the executable to study how the dimensionality of the convolutional layer changes, as a function of the supplied arguments.</p>
<p>Run <code>python path/to/conv-dims.py</code> and look at the output. It should appear as follows:</p>
</div>

</div>
<div class="divider"></div><div class="ud-atom">
  <h3></h3>
  <div>
  <figure class="figure">
    <img src="img/conv-dims.png" alt="" class="img img-fluid">
    <figcaption class="figure-caption">
      
    </figcaption>
  </figure>
</div>


</div>
<div class="divider"></div><div class="ud-atom">
  <h3></h3>
  <div>
  <p>Do the dimensions of the convolutional layer line up with your expectations?  </p>
<p>Feel free to change the values assigned to the arguments (<code>filters</code>, <code>kernel_size</code>, etc) in your <code>conv-dims.py</code> file.  </p>
<p>Take note of  how the <strong>number of parameters</strong> in the convolutional layer changes. This corresponds to the value under <code>Param #</code> in the printed output.  In the figure above, the convolutional layer has <code>80</code> parameters.</p>
<p>Also notice how the <strong>shape</strong> of the convolutional layer changes.  This corresponds to the value under <code>Output Shape</code> in the printed output.  In the figure above, <code>None</code> corresponds to the batch size, and the convolutional layer has a height of <code>100</code>, width of <code>100</code>, and depth of <code>16</code>.</p>
</div>

</div>
<div class="divider"></div><div class="ud-atom">
  <h3></h3>
  <div>
  <h3 id="formula-number-of-parameters-in-a-convolutional-layer">Formula: Number of Parameters in a Convolutional Layer</h3>
<p>The number of parameters in a convolutional layer depends on the supplied values of <code>filters</code>, <code>kernel_size</code>, and <code>input_shape</code>.  Let's define a few variables:</p>
<ul>
<li><code>K</code> - the number of filters in the convolutional layer </li>
<li><code>F</code> - the height and width of the convolutional filters</li>
<li><code>D_in</code> - the depth of the previous layer</li>
</ul>
<p>Notice that <code>K</code> = <code>filters</code>, and <code>F</code> = <code>kernel_size</code>.  Likewise, <code>D_in</code> is the last value in the <code>input_shape</code> tuple.</p>
<p>Since there are <code>F*F*D_in</code> weights per filter, and the convolutional layer is composed of <code>K</code> filters, the total number of weights in the convolutional layer is <code>K*F*F*D_in</code>.  Since there is one bias term per filter, the convolutional layer has <code>K</code> biases.  Thus, the <em>_ number of parameters_</em> in the convolutional layer is given by <code>K*F*F*D_in + K</code>.</p>
</div>

</div>
<div class="divider"></div><div class="ud-atom">
  <h3></h3>
  <div>
  <h3 id="formula-shape-of-a-convolutional-layer">Formula: Shape of a Convolutional Layer</h3>
<p>The shape of a convolutional layer depends on the supplied values of <code>kernel_size</code>, <code>input_shape</code>, <code>padding</code>, and <code>stride</code>.  Let's define a few variables:</p>
<ul>
<li><code>K</code> - the number of filters in the convolutional layer</li>
<li><code>F</code> - the height and width of the convolutional filters</li>
<li><code>S</code> - the stride of the convolution</li>
<li><code>H_in</code> - the height of the previous layer </li>
<li><code>W_in</code> - the width of the previous layer</li>
</ul>
<p>Notice that <code>K</code> = <code>filters</code>, <code>F</code> = <code>kernel_size</code>, and <code>S</code> = <code>stride</code>.  Likewise, <code>H_in</code> and <code>W_in</code> are the first and second value of the <code>input_shape</code> tuple, respectively.</p>
<p>The <strong>depth</strong> of the convolutional layer will always equal the number of filters <code>K</code>. </p>
<p>If <code>padding = 'same'</code>, then the spatial dimensions of the convolutional layer are the following:</p>
<ul>
<li><strong>height</strong> = ceil(float(<code>H_in</code>) / float(<code>S</code>))</li>
<li><strong>width</strong> = ceil(float(<code>W_in</code>) / float(<code>S</code>))</li>
</ul>
<p>If <code>padding = 'valid'</code>, then the spatial dimensions of the convolutional layer are the following:</p>
<ul>
<li><strong>height</strong> = ceil(float(<code>H_in</code> - <code>F</code> + 1) / float(<code>S</code>))</li>
<li><strong>width</strong> = ceil(float(<code>W_in</code> - <code>F</code> + 1) / float(<code>S</code>))</li>
</ul>
</div>

</div>
