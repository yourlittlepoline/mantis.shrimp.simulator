import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import colour
import matplotlib.pyplot as plt

st.title("Mantis shrimp vision simulator🦐")
data = pd.read_csv("mantis_shrimp_figure1A_digitized.csv")
wavelength_nm = data["wavelength_nm"]
receptor = data.iloc[:,1:]
receptor_matrix=receptor.values

upload_file=st.file_uploader("Upload your image (only png,jpeg,jpg)",
                             type=["png","jpg","jpeg"])

if upload_file is not None:
    image_pil=Image.open(upload_file).convert("RGB")
    image=np.array(image_pil)/255
    st.write(image.shape)
    fig6=plt.figure()
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")
    st.pyplot(fig6)

    @st.cache_data
    def calculate_receptors(image):
        height, width, channels = image.shape
        optimized_image=image.reshape(height*width,channels)
        unique_colors,inverse_indices=np.unique(optimized_image,axis=0,return_inverse=True)
        optimized_receptors=np.zeros((unique_colors.shape[0],11))
        optimized_spectra=np.zeros((unique_colors.shape[0],81))
        target_shape = colour.SpectralShape(300, 700, 5)

        for i in range (unique_colors.shape[0]):
            current_color=unique_colors[i]
            spectrum_test_pixel=colour.recovery.RGB_to_sd_Smits1999(np.array(current_color))
            test_spectrum_align=spectrum_test_pixel.copy().align(target_shape)
            try_spectrum=test_spectrum_align.values.copy()
            try_spectrum[0:16]=0
            result=try_spectrum@receptor_matrix
            optimized_receptors[i,:]=result
            optimized_spectra[i,:]=try_spectrum

        rebuild=optimized_receptors[inverse_indices].reshape(height,width,11)
        spectral_image=optimized_spectra[inverse_indices].reshape(height,width,81)
        return rebuild, spectral_image
    
    rebuild, spectral_image = calculate_receptors(image)

    st.write(rebuild.shape)
    st.write(spectral_image.shape)

    contrast_horizontal=np.sum(
    (rebuild[:,0:-1]-rebuild[:,1:])**2,
    axis=2
    )**0.5

    contrast_vertical=np.sum(
    (rebuild[0:-1,:]-rebuild[1:,:])**2,
    axis=2
    )**0.5

    contrast=((contrast_vertical[:,0:-1]**2)+(contrast_horizontal[0:-1,:]**2))**0.5

    fig=plt.figure()
    plt.imshow(contrast_horizontal)
    plt.title("Horizontal contrasts")
    plt.axis("off")
    st.pyplot(fig)

    fig2=plt.figure()
    plt.imshow(contrast_vertical)
    plt.title("Vertical contrasts")
    plt.axis("off")
    st.pyplot(fig2)

    fig3=plt.figure()
    plt.imshow(contrast)
    plt.title("Full contrast")
    plt.axis("off")
    st.pyplot(fig3)

    comparison_data=pd.read_csv("mantis_behavioral_euclidean_comparison.csv")

    reference=comparison_data.iloc[:,0]
    behavioral_thresholds=comparison_data.iloc[:,2]
    wl_to_process=np.arange(300,701,5)
    interpolated_thresholds=np.interp(wl_to_process,reference,behavioral_thresholds)
    fig4=plt.figure()
    plt.plot(wl_to_process,interpolated_thresholds)
    plt.xlim(400,650)
    plt.title("Behavioral Color Discrimination Thresholds")
    plt.xlabel("Reference Wavelength (nm)")
    plt.ylabel("Smallest Tested Significant Difference (nm)")
    st.pyplot(fig4)

    D_significant=comparison_data.iloc[:,6]
    D_thresholds=np.interp(wl_to_process,reference,D_significant)
    peak_wavelength_indices=np.argmax(spectral_image,axis=2)

    pixel_thresholds=D_thresholds[peak_wavelength_indices]

    visible_edges=contrast>=pixel_thresholds[0:-1,0:-1]
    fig5=plt.figure()
    plt.imshow(visible_edges)

    plt.title("Heuristic Behavioral Contrast Mask")
    plt.axis("off")
    st.pyplot(fig5)

else:
    st.write("There is no picture")

