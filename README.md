# CGBA: Curvature-aware Geometric Black-box Attack
Welcome to our repository featuring the official implementation of the CGBA algorithm, as published in ICCV2023.
## Requirements
Before executing the code, ensure that the following packages are installed in your environment:
* PyTorch and Trochvision
* Numpy
* Os
* SciPy
  
Or you can type the following to create an environment:  

```html
<div class="copy-container">
  <code>
    function greet(name) {
      return "Hello, " + name + "!";
    }
  </code>
  <button class="copy-button" onclick="copyToClipboard()">
    Copy
  </button>
</div>

<script>
function copyToClipboard() {
  const codeBlock = document.querySelector('.copy-container code');
  const textArea = document.createElement('textarea');
  textArea.value = codeBlock.textContent;
  document.body.appendChild(textArea);
  textArea.select();
  document.execCommand('copy');
  document.body.removeChild(textArea);
}
</script>

<style>
.copy-container {
  position: relative;
}

.copy-button {
  position: absolute;
  top: 5px;
  right: 5px;
  font-size: 12px;
  cursor: pointer;
}
</style>

