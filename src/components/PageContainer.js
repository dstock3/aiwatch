import React from 'react'
import Navigation from './sections/Navigation'
const PageContainer = ({ Page }) => {
  return (
    <>
      <Navigation />
      <Page />
      
    </>
  )
}

export default PageContainer