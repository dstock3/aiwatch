import React from 'react'
import Footer from './sections/Footer'
import Navigation from './sections/Navigation'
import '../App.css'

const PageContainer = ({ Page }) => {
  return (
    <>
      <Navigation />
      <main className="main-content">
        <Page />
      </main>
      <Footer />
    </>
  );
};

export default PageContainer