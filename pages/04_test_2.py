import streamlit as st

def page_1():
    st.write("This is Home")

def page_2():
    st.write("This is Unsupervised Clusters")

def page_3():
    st.write("This is Supervised Classifiers")
    
def page_4():
    st.write("This is Time Series Analysis")  

def page_5():
    st.write("This is Algorithmic Trading")  

def main():
    st.set_page_config(page_title="My App", page_icon=":guardsman:", layout="wide")
    st.markdown("""
    <style>
    .nav-link {
    padding : 10px;
    margin: 10px;
    text-decoration: none;
    font-size: 20px;
    color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)
    # Create the navigation bar
    st.header("Stock analyzer & prediction App")
    st.markdown(
        '''<a href="#page-1" class="nav-link">Home</a> 
        <a href="#page-2" class="nav-link">Unsupervised Clusters</a> 
        <a href="#page-3" class="nav-link">Supervised Classifiers</a>
        <a href="#page-4" class="nav-link">Time Series</a>
        <a href="#page-5" class="nav-link">Algo Trading</a>''',
        unsafe_allow_html=True,
    )

    # Show the appropriate page
    if st.checkbox("1"):
        st.subheader("Hello 1") 
        page_1()
    if st.checkbox("2"):
        st.subheader("Hello 2")
        page_2()
    if st.checkbox("3"):
        st.subheader("Hello 3")
        page_3()

if __name__ == "__main__":
    main()




